#if !defined(CLANGD) && !defined(__CUDACC__)
#include <cilk/cilk.h>
#include <cilk/cilkscale.h>
#else
#include <cstdint>
#define cilk_for for
#define cilk_spawn
#define cilk_sync
#define cilk_scope
#define cilk_reducer(zero_fn, add_fn)
#endif

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

struct GoldilockField {
    constexpr static uint64_t PRIME = 0xffffffff00000001ull;

    // See https://xn--2-umb.com/22/goldilocks/#a-lucky-binary-base
    constexpr static uint64_t GENERATOR = 2717;

    // PRIME - 1 has MULTIPLICITY_2 powers of 2
    constexpr static uint64_t MULTIPLICITY_2 = 32;

    // NOLINTNEXTLINE(*-type-member-init, *-use-equals-default)
    GoldilockField() { /* Deliberately skip initialization of _raw
                         to allow more aggresive optimization */
    }

    explicit GoldilockField(uint64_t raw) : _raw{raw} {}

    explicit operator uint64_t() const { return _raw; }

    auto operator+(GoldilockField other) const -> GoldilockField {
        const uint64_t sum = _raw + other._raw;
        const GoldilockField ret{
            sum < _raw || sum < other._raw || sum >= PRIME ? sum - PRIME : sum};
        return ret;
    }

    auto operator-(GoldilockField other) const -> GoldilockField {
        const uint64_t diff = _raw - other._raw;
        const GoldilockField ret{(diff > _raw) ? diff + PRIME : diff};
        return ret;
    }

    auto operator*(GoldilockField other) const -> GoldilockField {
        // Start by carrying out an ordinary 64x64->128 bit multiplication
        const uint32_t a0 = _raw, a1 = _raw >> 32;
        const uint32_t b0 = other._raw, b1 = other._raw >> 32;
        const uint64_t p0 = static_cast<uint64_t>(a0) * b0,
                       p1 = static_cast<uint64_t>(a0) * b1,
                       p2 = static_cast<uint64_t>(a1) * b0,
                       p3 = static_cast<uint64_t>(a1) * b1;
        const uint32_t cy = ((p0 >> 32) + static_cast<uint32_t>(p1) +
                             static_cast<uint32_t>(p2)) >>
                            32;
        const uint64_t x = p0 + (p1 << 32) + (p2 << 32),
                       y = p3 + (p1 >> 32) + (p2 >> 32) + cy;
        // Store result in 4 32-bit words
        const uint32_t c0 = x, c1 = x >> 32, c2 = y, c3 = y >> 32;

        // Now perform reduction: modulus is phi^2 - phi + 1 where phi = 2^32
        //   ab = c0 + c1*phi + c2*phi^2 + c3*phi^3
        // Exploit phi^2 = phi-1 and phi^3 = phi * (phi-1) = (phi-1) - phi = -1
        //   ab = c0 + c1*phi + c2*(phi-1) - c3
        //      = (c0-c2-c3) + (c1+c2)*phi
        const GoldilockField ret =
            (GoldilockField(c0) - GoldilockField(c2) - GoldilockField(c3)) +
            (GoldilockField(static_cast<uint64_t>(c1) << 32) +
             GoldilockField(static_cast<uint64_t>(c2) << 32));
        return ret;
    }

    auto operator==(GoldilockField other) const -> bool {
        return _raw == other._raw;
    }

    auto operator!=(GoldilockField other) const -> bool {
        return _raw != other._raw;
    }

    // No comparison operators as they make no sense for finite fields

    auto operator+=(GoldilockField other) -> GoldilockField & {
        return *this = *this + other;
    }

    auto operator-=(GoldilockField other) -> GoldilockField & {
        return *this = *this - other;
    }

    auto operator*=(GoldilockField other) -> GoldilockField & {
        return *this = *this * other;
    }

    template <typename RNG>
    inline static auto random(RNG &rng) -> GoldilockField {
        static std::uniform_int_distribution<uint64_t> dist(0, PRIME - 1);
        return GoldilockField{dist(rng)};
    }

  private:
    uint64_t _raw;
};

template <> struct std::formatter<GoldilockField> : std::formatter<uint64_t> {
    template <typename FormatContext>
    auto format(GoldilockField t, FormatContext &ctx) const {
        return std::formatter<uint64_t>::format(static_cast<uint64_t>(t), ctx);
    }
};

static inline auto operator<<(std::ostream &os, const GoldilockField &s)
    -> std::ostream & {
    return os << static_cast<uint64_t>(s);
}

using Scalar = GoldilockField;

struct SparseMatEntry {
    uint32_t row{};
    uint32_t col{};
    Scalar val;
};

struct SparseMatPoly {
    uint32_t n_rows;
    uint32_t n_cols;
    std::vector<SparseMatEntry> entries;
};

struct R1CSWitness;

struct R1CS {
    /// Number of constraints
    uint32_t n_cons{};
    /// Number of private inputs & wire values
    uint32_t n_vars{};
    /// Number of public inputs & outputs
    uint32_t n_pub_io{};

    SparseMatPoly A;
    SparseMatPoly B;
    SparseMatPoly C;

    auto n_wires() const -> uint32_t { return n_vars + 1 + n_pub_io; }

    static auto from_file(const std::filesystem::path &path) -> R1CS;

    auto is_valid_witness(const R1CSWitness &witness) const -> bool;
};

struct R1CSWitness {
    std::vector<Scalar> z;

    static auto from_file(const std::filesystem::path &path,
                          const R1CS &instance) -> R1CSWitness;
};

namespace {

template <typename T> auto read_le(std::string_view &view) -> T {
    // THIS ONLY WORKS ON LITTLE ENDIAN MACHINES
    if (view.length() < sizeof(T)) {
        throw std::invalid_argument{
            "invalid R1CS file: unexpected EOF while reading data"};
    }
    T value;
    std::memcpy(&value, view.data(), sizeof(T));
    view = view.substr(sizeof(T));
    return value;
}

/**
 * @brief Read a binary file and check the magic number and version
 * @param path The path to the file
 * @param format The expected magic number, also double as format name
                 (as is the case for r1cs and wtns)
 * @param version The expected version number
 * @returns The header section and the content section
 */
auto read_binary_file(const std::filesystem::path &path,
                      const std::string format, const uint32_t version)
    -> std::pair<std::string, std::string> {

    std::ifstream file{path};
    const std::string content{std::istreambuf_iterator<char>{file}, {}};

    std::string_view view{content};

    // Magic number check
    const auto magic_number_ = view.substr(0, 4);
    view = view.substr(4);
    if (magic_number_ != format) {
        throw std::invalid_argument{
            std::format("invalid {} file: magic number mismatch, found {}",
                        format, magic_number_)};
    }

    const auto version_ = read_le<uint32_t>(view);
    if (version_ != version) {
        throw std::invalid_argument{std::format(
            "invalid {} file: unsupported version {}", format, version_)};
    }

    const auto n_sections = read_le<uint32_t>(view);

    std::string_view header_section, content_section;

    for (uint32_t i = 0; i < n_sections; i++) {
        const auto section_type = read_le<uint32_t>(view);
        const auto section_size = read_le<uint64_t>(view);

        switch (section_type) {
        case 1:
            header_section = view.substr(0, section_size);
            break;
        case 2:
            content_section = view.substr(0, section_size);
            break;
        default:
            // Ignore other sections
            ;
        }
        view = view.substr(section_size);
    }

    if (header_section.empty() || content.empty()) {
        throw std::invalid_argument{std::format(
            "invalid {} file: missing header / content section", format)};
    }
    return {std::string{header_section}, std::string{content_section}};
}

void check_header_prime(std::string_view &header_section,
                        const std::string &format) {
    const auto fs = read_le<uint32_t>(header_section);
    if (fs != sizeof(Scalar)) {
        throw std::invalid_argument{std::format(
            "invalid {} file: unsupported field size {}", format, fs)};
    }
    // Note for future self
    static_assert(std::is_same_v<Scalar, GoldilockField>,
                  "need to review prime-checking logic for other fields");
    const auto prime = read_le<uint64_t>(header_section);
    if (prime != Scalar::PRIME) {
        throw std::invalid_argument{std::format(
            "invalid {} file: unsupported prime {}", format, prime)};
    }
}

/**
 * Remaps column index / repermutes column.
 *
 * R1CS file format assumes the following ordering of variables:
 *     1, public outputs, public inputs, private inputs, wire values.
 *
 * Whereas the Spartan paper assumes the following ordering:
 *     var, 1, io
 * where var includes private inputs + wire values, and io includes public
 * inputs and outputs.
 */
auto remap_col(uint32_t col, const R1CS &instance) -> uint32_t {
    if (col <= instance.n_pub_io) {
        return col + instance.n_vars;
    } else {
        return col - instance.n_pub_io - 1;
    }
}

auto compute_spmv(const SparseMatPoly &mat, const std::vector<Scalar> &z)
    -> std::vector<Scalar> {
    std::vector<std::atomic<Scalar>> temp(mat.n_rows);
    cilk_for (size_t i = 0; i < mat.entries.size(); i++) {
        const Scalar prod = mat.entries[i].val * z[mat.entries[i].col];
        auto &t = temp[mat.entries[i].row];
        for (Scalar acc = t.load(std::memory_order_acquire);
             !t.compare_exchange_weak(acc, acc + prod,
                                      std::memory_order_acq_rel,
                                      std::memory_order_release);)
            ;
    }
    std::vector<Scalar> result(mat.n_rows);
    for (size_t i = 0; i < mat.n_rows; i++) {
        result[i] = temp[i].load(std::memory_order_acquire);
    }
    return result;
}

// Do not inline
__attribute__((noinline)) auto compute_spmvs(const R1CS &r1cs,
                                             const R1CSWitness &witness)
    -> std::tuple<std::vector<Scalar>, std::vector<Scalar>,
                  std::vector<Scalar>> {

    std::vector<Scalar> Az, Bz, Cz;

    cilk_scope {
        cilk_spawn { Az = compute_spmv(r1cs.A, witness.z); }
        cilk_spawn { Bz = compute_spmv(r1cs.B, witness.z); }
        Cz = compute_spmv(r1cs.C, witness.z);
    }

    return {Az, Bz, Cz};
}

} // namespace

auto R1CS::from_file(const std::filesystem::path &path) -> R1CS {
    // Reference:
    // https://github.com/iden3/r1csfile/blob/master/doc/r1cs_bin_format.md

    const auto [header_section_, constr_section_] =
        read_binary_file(path, "r1cs", 1);
    std::string_view header_section{header_section_};
    std::string_view constr_section{constr_section_};

    check_header_prime(header_section, "r1cs");
    const auto n_wires = read_le<uint32_t>(header_section);
    const auto n_pub_out = read_le<uint32_t>(header_section);
    const auto n_pub_in = read_le<uint32_t>(header_section);
    const auto n_prv_in = read_le<uint32_t>(header_section);
    const auto n_labels = read_le<uint64_t>(header_section);
    const auto m_constraints = read_le<uint32_t>(header_section);

    std::cout << std::format("n_wires {} n_pub_out {} n_pub_in {} "
                             "n_prv_in {} n_labels {} m_constraints {}",
                             n_wires, n_pub_out, n_pub_in, n_prv_in, n_labels,
                             m_constraints)
              << std::endl;

    R1CS r1cs;
    r1cs.A.n_rows = m_constraints;
    r1cs.A.n_cols = n_wires;
    r1cs.B.n_rows = m_constraints;
    r1cs.B.n_cols = n_wires;
    r1cs.C.n_rows = m_constraints;
    r1cs.C.n_cols = n_wires;
    r1cs.n_pub_io = n_pub_out + n_pub_in;
    r1cs.n_vars = n_wires - r1cs.n_pub_io - 1;
    r1cs.n_cons = m_constraints;

    for (uint32_t row = 0; row < m_constraints; row++) {
        const auto n_A = read_le<uint32_t>(constr_section);
        for (uint32_t i = 0; i < n_A; i++) {
            const auto col = read_le<uint32_t>(constr_section);
            const auto val = read_le<Scalar>(constr_section);
            r1cs.A.entries.push_back({row, remap_col(col, r1cs), val});
        }
        const auto n_B = read_le<uint32_t>(constr_section);
        for (uint32_t i = 0; i < n_B; i++) {
            const auto col = read_le<uint32_t>(constr_section);
            const auto val = read_le<Scalar>(constr_section);
            r1cs.B.entries.push_back({row, remap_col(col, r1cs), val});
        }
        const auto n_C = read_le<uint32_t>(constr_section);
        for (uint32_t i = 0; i < n_C; i++) {
            const auto col = read_le<uint32_t>(constr_section);
            const auto val = read_le<Scalar>(constr_section);
            r1cs.C.entries.push_back({row, remap_col(col, r1cs), val});
        }
    }

    return r1cs;
}

auto R1CSWitness::from_file(const std::filesystem::path &path,
                            const R1CS &instance) -> R1CSWitness {
    // Reference: https://github.com/iden3/snarkjs/blob/master/src/wtns_utils.js

    const auto [header_section_, witness_section_] =
        read_binary_file(path, "wtns", 2);
    std::string_view header_section{header_section_};
    std::string_view witness_section{witness_section_};

    check_header_prime(header_section, "wtns");
    const auto n_witness = read_le<uint32_t>(header_section);

    if (n_witness != instance.n_wires()) {
        throw std::invalid_argument{std::format(
            "invalid wtns file: witness size mismatch, expected {} got {}",
            instance.n_vars, n_witness)};
    }

    R1CSWitness witness;
    witness.z.resize(n_witness);
    for (uint32_t i = 0; i < n_witness; i++) {
        const auto val = read_le<Scalar>(witness_section);
        witness.z[remap_col(i, instance)] = val;
    }
    std::cout << std::format("n_witness {}", n_witness) << std::endl;

    return witness;
}

auto main() -> int {
    const auto r1cs = R1CS::from_file(
        "/home/chengyuan/Projects/meng/gpu/circuits/sha256.r1cs");
    const auto witness = R1CSWitness::from_file(
        "/home/chengyuan/Projects/meng/gpu/circuits/sha256.wtns", r1cs);

    const auto [Az, Bz, Cz] = compute_spmvs(r1cs, witness);
    for (size_t i = 0; i < r1cs.n_cons; i++) {
        if (Az[i] * Bz[i] != Cz[i]) {
            std::cout << std::format(
                             "Az[{}] * Bz[{}] = {} * {} = {} =?= {} = Cz[{}]",
                             i, i, Az[i], Bz[i], Az[i] * Bz[i], Cz[i], i)
                      << std::endl;
        }
    }
    return 0;
}