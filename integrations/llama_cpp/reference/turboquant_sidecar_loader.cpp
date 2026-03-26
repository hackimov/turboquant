// SPDX-License-Identifier: Apache-2.0
// Reference loader + CPU dequant for TurboQuant sidecar (format v1).
// Not linked into llama.cpp; copy/adapt under upstream license if needed.
//
// Binary layout must match turboquant.llama_cpp_pack.serialize_quantizer_metadata:
//   v1 header 32 bytes LE: magic[8]=="TURBOQT1", uint32 ver==1,
//                           uint32 bits, uint32 head_dim, uint32 k_centroids, double qjl_factor
//   v2 header 40 bytes LE: magic[8]=="TURBOQT1", uint32 ver==2,
//                           float64 bits, uint32 head_dim, uint32 k_centroids, uint32 codebook_id, double qjl_factor
//   payload: float32 centroids[k], float32 Pi[d*d] row-major, float32 S[d*d] row-major

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace turboquant_ref {

static constexpr char kMagic[8] = {'T', 'U', 'R', 'B', 'O', 'Q', 'T', '1'};

struct MetaV1 {
    double bits = 0.0;
    int head_dim = 0;
    int k_centroids = 0;
    int codebook_id = 0;
    double qjl_factor = 0.0;
    std::vector<float> centroids;
    std::vector<float> pi; // row-major d x d
    std::vector<float> s;  // row-major d x d
};

inline void read_u32(const uint8_t * p, uint32_t & o) {
    std::memcpy(&o, p, 4);
}

inline void read_f64(const uint8_t * p, double & o) {
    std::memcpy(&o, p, 8);
}

inline MetaV1 load_meta_v1(const std::vector<uint8_t> & blob) {
    if (blob.size() < 32) {
        throw std::runtime_error("turboquant meta: truncated header");
    }
    if (std::memcmp(blob.data(), kMagic, 8) != 0) {
        throw std::runtime_error("turboquant meta: bad magic");
    }
    uint32_t ver = 0;
    read_u32(blob.data() + 8, ver);
    uint32_t d = 0, k = 0;
    double bits = 0.0;
    int codebook_id = 0;
    double qjl = 0.0;
    size_t hlen = 0;

    if (ver == 1) {
        uint32_t bits_i = 0;
        read_u32(blob.data() + 12, bits_i);
        bits = double(bits_i);
        read_u32(blob.data() + 16, d);
        read_u32(blob.data() + 20, k);
        read_f64(blob.data() + 24, qjl);
        codebook_id = 0; // inferred in Python; legacy sidecar used paper/ternary inference.
        hlen = 32;
    } else if (ver == 2) {
        read_f64(blob.data() + 12, bits);
        read_u32(blob.data() + 20, d);
        read_u32(blob.data() + 24, k);
        uint32_t cbid_u32 = 0;
        read_u32(blob.data() + 28, cbid_u32);
        codebook_id = int(cbid_u32);
        read_f64(blob.data() + 32, qjl);
        hlen = 40;
    } else {
        throw std::runtime_error("turboquant meta: unsupported version");
    }

    const size_t need = hlen + size_t(k) * 4u + 2u * size_t(d) * size_t(d) * 4u;
    if (blob.size() < need) {
        throw std::runtime_error("turboquant meta: truncated payload");
    }

    MetaV1 m;
    m.bits = bits;
    m.head_dim = int(d);
    m.k_centroids = int(k);
    m.codebook_id = codebook_id;
    m.qjl_factor = qjl;
    m.centroids.resize(k);
    m.pi.resize(size_t(d) * size_t(d));
    m.s.resize(size_t(d) * size_t(d));

    size_t off = hlen;
    std::memcpy(m.centroids.data(), blob.data() + off, k * 4u);
    off += k * 4u;
    std::memcpy(m.pi.data(), blob.data() + off, size_t(d) * size_t(d) * 4u);
    off += size_t(d) * size_t(d) * 4u;
    std::memcpy(m.s.data(), blob.data() + off, size_t(d) * size_t(d) * 4u);

    const double expected_qjl = std::sqrt(3.14159265358979323846 / 2.0) / double(d);
    if (std::abs(qjl - expected_qjl) > 1e-5) {
        throw std::runtime_error("turboquant meta: qjl_factor mismatch");
    }
    return m;
}

// DeQuantprod on one unit row: idx[j] in [0, k), qjl_sign[j] in {-1, +1}, scalar gamma.
// Output unit vector x_tilde[d]; multiply by x_norm outside for full K/V row.
inline void dequant_unit_row(
    const MetaV1 & m,
    const int64_t * idx,          // [d]
    const float * qjl_sign,       // [d], values should be -1.f or +1.f
    float gamma,
    float * x_tilde_out           // [d]
) {
    const int d = m.head_dim;
    // x_mse_unit = (centroids[idx] @ Pi)  — rows: y_tilde = centroids per coord, then x = y @ Pi
    std::vector<float> y_tilde(size_t(d));
    for (int j = 0; j < d; ++j) {
        int64_t t = idx[j];
        if (t < 0 || t >= m.k_centroids) {
            throw std::runtime_error("idx out of range");
        }
        y_tilde[static_cast<size_t>(j)] = m.centroids[static_cast<size_t>(t)];
    }
    // x_mse = y_tilde * Pi  (row vector times matrix, Pi row-major dxd)
    for (int i = 0; i < d; ++i) {
        float s = 0.f;
        for (int j = 0; j < d; ++j) {
            s += y_tilde[static_cast<size_t>(j)] * m.pi[static_cast<size_t>(j) * size_t(d) + size_t(i)];
        }
        x_tilde_out[i] = s;
    }
    // qjl: x_qjl = qjl_factor * gamma * (qjl_sign @ S)  — row times S, S row-major
    for (int i = 0; i < d; ++i) {
        float acc = 0.f;
        for (int j = 0; j < d; ++j) {
            acc += qjl_sign[j] * m.s[static_cast<size_t>(j) * size_t(d) + size_t(i)];
        }
        x_tilde_out[i] += float(m.qjl_factor) * gamma * acc;
    }
}

inline void dequant_full_row(
    const MetaV1 & m,
    const int64_t * idx,
    float x_norm,
    const float * qjl_sign,
    float gamma,
    float * x_out
) {
    dequant_unit_row(m, idx, qjl_sign, gamma, x_out);
    const int d = m.head_dim;
    for (int i = 0; i < d; ++i) {
        x_out[i] *= x_norm;
    }
}

} // namespace turboquant_ref
