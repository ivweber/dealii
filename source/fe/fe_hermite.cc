/**
 * Source code for Hermite basis functions.
 */

#include <deal.II/base/template_constraints.h>

#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/fe_tools.h>

#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  inline std::vector<unsigned int>
  get_hermite_dpo_vector(const unsigned int dim,
                         const unsigned int regularity,
                         const unsigned int nodes)
  {
    std::vector<unsigned int> result(dim + 1);
    unsigned int coeff1 = 1, coeff2 = Utilities::pow(regularity + 1, dim);
    for (unsigned int i = 0; i <= dim; ++i)
      {
        result[i] = coeff1 * coeff2;
        coeff1 *= nodes;
        coeff2 /= regularity + 1;
      }
    return result;
  }

  template <int dim>
  void
  hermite_hierarchic_to_lexicographic_numbering(const unsigned int regularity,
                                                const unsigned int nodes,
                                                std::vector<unsigned int> &h2l)
  {
    const unsigned int node_dofs_1d = regularity + 1;
    const unsigned int node_dofs_2d = node_dofs_1d * node_dofs_1d;
    const unsigned int node_dofs_3d = node_dofs_2d * node_dofs_1d;
    const unsigned int dim_dofs_1d  = 2 * node_dofs_1d + nodes;
    const unsigned int dim_dofs_2d  = dim_dofs_1d * dim_dofs_1d;
    const unsigned int dim_dofs_3d  = dim_dofs_2d * dim_dofs_1d;
    unsigned int       offset, count = 0;
    AssertDimension(h2l.size(), Utilities::pow(node_dofs_1d, dim));
    switch (dim)
      {
        case 1:
          // Assign DOFs at nodes
          for (unsigned int i = 0; i < node_dofs_1d; ++i, ++count)
            h2l[i] = i;
          for (unsigned int i = 0; i < node_dofs_1d; i++, count++)
            h2l[i + node_dofs_1d] = i + node_dofs_1d + nodes;
          // Assign DOFs on line if needed
          for (unsigned int i = 0; i < nodes; i++, count++)
            h2l[i + 2 * node_dofs_1d] = i + node_dofs_1d;
          AssertDimension(count, Utilities::pow(node_dofs_1d, dim));
          break;
        case 2:
          // Assign DOFs at nodes
          for (unsigned int i = 0; i < node_dofs_1d; i++)
            for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
              h2l[j + i * node_dofs_1d] = j + i * dim_dofs_1d;
          offset = node_dofs_1d * node_dofs_1d;
          for (unsigned int i = 0; i < node_dofs_1d; i++)
            for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
              h2l[j + i * node_dofs_1d + offset] =
                j + node_dofs_1d + nodes + i * dim_dofs_1d;
          offset *= 2;
          for (unsigned int i = 0; i < node_dofs_1d; i++)
            for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
              h2l[j + i * node_dofs_1d + offset] =
                j + (i + node_dofs_1d + nodes) * dim_dofs_1d;
          offset += node_dofs_1d * node_dofs_1d;
          for (unsigned int i = 0; i < node_dofs_1d; i++)
            for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
              h2l[j + i * node_dofs_1d + offset] =
                j + (node_dofs_1d + nodes) * (dim_dofs_1d + 1) +
                i * dim_dofs_1d;
          if (nodes)
            {
              offset += node_dofs_1d * node_dofs_1d;
              // Assign DOFs on edges
              for (unsigned int i = 0; i < nodes; i++)
                {
                  for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
                    h2l[j + 2 * i * node_dofs_1d + offset] =
                      j + (i + node_dofs_1d) * dim_dofs_1d;
                  for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
                    h2l[j + (2 * i + 1) * node_dofs_1d + offset] =
                      j + (i + node_dofs_1d) * dim_dofs_1d + node_dofs_1d +
                      nodes;
                }
              offset += 2 * nodes * node_dofs_1d;
              for (unsigned int i = 0; i < nodes; i++)
                {
                  for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
                    h2l[j + 2 * i * node_dofs_1d + offset] =
                      i + j * dim_dofs_1d + node_dofs_1d;
                  for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
                    h2l[j + (2 * i + 1) * node_dofs_1d + offset] =
                      i + (j + node_dofs_1d + nodes) * dim_dofs_1d +
                      node_dofs_1d;
                }
              offset += 2 * nodes * node_dofs_1d;
              // Assign DOFs on face
              for (unsigned int i = 0; i < nodes; i++)
                for (unsigned int j = 0; j < nodes; j++, count++)
                  h2l[j + i * nodes + offset] =
                    j + (i + node_dofs_1d) * dim_dofs_1d + node_dofs_1d;
            }
          AssertDimension(count, Utilities::pow(node_dofs_1d, dim));
          break;
        case 3:
          // Assign DOFs at nodes
          offset = 0;
          for (unsigned int di = 0; di < 2; di++)
            for (unsigned int dj = 0; dj < 2; dj++)
              for (unsigned int dk = 0; dk < 2; dk++)
                {
                  for (unsigned int i = 0; i < node_dofs_1d; i++)
                    for (unsigned int j = 0; j < node_dofs_1d; j++)
                      for (unsigned int k = 0; k < node_dofs_1d; k++, count++)
                        h2l[k + j * node_dofs_1d + i * node_dofs_2d + offset] =
                          k + j * dim_dofs_1d + i * dim_dofs_2d +
                          (node_dofs_1d + nodes) *
                            (dk + dj * dim_dofs_1d + di * dim_dofs_2d);
                  offset += node_dofs_1d * node_dofs_2d;
                }
          if (nodes)
            {
              // Assign DOFs on edges
              // edges parallel to z
              for (unsigned int i = 0; i < nodes; i++)
                for (unsigned int dj = 0; dj < 2; dj++)
                  for (unsigned int dk = 0; dk < 2; dk++)
                    {
                      for (unsigned int j = 0; j < node_dofs_1d; j++)
                        for (unsigned int k = 0; k < node_dofs_1d; k++, count++)
                          h2l[k + j * node_dofs_1d + offset] =
                            k + j * dim_dofs_1d +
                            (i + node_dofs_1d) * dim_dofs_2d +
                            (node_dofs_1d + nodes) * (dk + dj * dim_dofs_1d);
                      offset += node_dofs_2d;
                    }
              // edges parallel to y
              for (unsigned int j = 0; j < nodes; j++)
                for (unsigned int di = 0; di < 2; di++)
                  for (unsigned int dk = 0; dk < 2; dk++)
                    {
                      for (unsigned int i = 0; i < node_dofs_1d; i++)
                        for (unsigned int k = 0; k < node_dofs_1d; k++, count++)
                          h2l[k + i * node_dofs_1d + offset] =
                            k + i * dim_dofs_2d +
                            (j + node_dofs_1d) * dim_dofs_1d +
                            (node_dofs_1d + nodes) * (dk + di * dim_dofs_2d);
                      offset += node_dofs_2d;
                    }
              // edges parallel to x
              for (unsigned int k = 0; k < nodes; k++)
                for (unsigned int di = 0; di < 2; di++)
                  for (unsigned int dj = 0; dj < 2; dj++)
                    {
                      for (unsigned int i = 0; i < node_dofs_1d; i++)
                        for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
                          h2l[j + i * node_dofs_1d + offset] =
                            j * dim_dofs_1d + i * dim_dofs_2d + k +
                            node_dofs_1d +
                            (node_dofs_1d + nodes) *
                              (dj * dim_dofs_1d + di * dim_dofs_2d);
                      offset += node_dofs_2d;
                    }
              // Assign DOFs on faces
              // faces normal to x
              for (unsigned int i = 0; i < nodes; i++)
                for (unsigned int j = 0; j < nodes; j++)
                  for (unsigned int dk = 0; dk < 2; dk++)
                    {
                      for (unsigned int k = 0; k < node_dofs_1d; k++, count++)
                        h2l[k + offset] = k + (j + node_dofs_1d) * dim_dofs_1d +
                                          (i + node_dofs_1d) * dim_dofs_2d +
                                          (node_dofs_1d + nodes) * dk;
                      offset += node_dofs_1d;
                    }
              // faces normal to y
              for (unsigned int i = 0; i < nodes; i++)
                for (unsigned int k = 0; k < nodes; k++)
                  for (unsigned int dj = 0; dj < 2; dj++)
                    {
                      for (unsigned int j = 0; j < node_dofs_1d; j++, count++)
                        h2l[j + offset] =
                          j * dim_dofs_1d + k + node_dofs_1d +
                          (i + node_dofs_1d) * dim_dofs_2d +
                          (node_dofs_1d + nodes) * dj * dim_dofs_1d;
                      offset += node_dofs_1d;
                    }
              // faces normal to z
              for (unsigned int j = 0; j < nodes; j++)
                for (unsigned int k = 0; k < nodes; k++)
                  for (unsigned int di = 0; di < 2; di++)
                    {
                      for (unsigned int i = 0; i < node_dofs_1d; i++, count++)
                        h2l[i + offset] =
                          i * dim_dofs_2d + k + node_dofs_1d +
                          (j + node_dofs_1d) * dim_dofs_1d +
                          (node_dofs_1d + nodes) * di * dim_dofs_2d;
                      offset += node_dofs_1d;
                    }
              // Assign DOFs in cell
              for (unsigned int i = 0; i < nodes; i++)
                for (unsigned int j = 0; j < nodes; j++)
                  for (unsigned int k = 0; k < nodes; k++, count++)
                    h2l[k + (j + i * nodes) * nodes + offset] =
                      k + node_dofs_1d + (j + node_dofs_1d) * dim_dofs_1d +
                      (i + node_dofs_1d) * dim_dofs_2d;
            }
          AssertDimension(count, Utilities::pow(node_dofs_1d, dim));
          break;
        case 4:
          // Assign DOFs at nodes
          offset = 0;
          for (unsigned int di = 0; di < 2; di++)
            for (unsigned int dj = 0; dj < 2; dj++)
              for (unsigned int dk = 0; dk < 2; dk++)
                for (unsigned int dl = 0; dl < 2; dl++)
                  {
                    for (unsigned int i = 0; i < node_dofs_1d; i++)
                      for (unsigned int j = 0; j < node_dofs_1d; j++)
                        for (unsigned int k = 0; k < node_dofs_1d; k++)
                          for (unsigned int l = 0; l < node_dofs_1d;
                               l++, count++)
                            h2l[l + k * node_dofs_1d + j * node_dofs_2d +
                                i * node_dofs_3d + offset] =
                              l + k * dim_dofs_1d + j * dim_dofs_2d +
                              i * dim_dofs_3d +
                              (node_dofs_1d + nodes) *
                                (dl + dk * dim_dofs_1d + dj * dim_dofs_2d +
                                 di * dim_dofs_3d);
                    offset += node_dofs_1d * node_dofs_3d;
                  }
          if (nodes)
            {
              Assert(false, ExcNotImplemented());
              // Assign DOFs on edges
              // edges parallel to w
              // edges parallel to z
              // edges parallel to y
              // edges parallel to x
              // Assign DOFs on 2D faces
              // faces normal to xy
              // faces normal to xz
              // faces normal to xw
              // faces normal to yz
              // faces normal to yw
              // faces normal to zw
              // Assign DOFs on 3D faces
              // faces normal to x
              // faces normal to y
              // faces normal to z
              // faces normal to w
              // Assign DOFs in cell
            }
          AssertDimension(count, Utilities::pow(node_dofs_1d, dim));
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
  }

  template <int dim>
  inline std::vector<unsigned int>
  hermite_face_lexicographic_to_hierarchic_numbering(
    const unsigned int regularity,
    const unsigned int nodes)
  {
    if (dim <= 1)
      return std::vector<unsigned int>();

    const std::vector<unsigned int> dpo =
      get_hermite_dpo_vector(dim - 1, regularity, nodes);
    const dealii::FiniteElementData<dim - 1> face_data(
      dpo, 1, 2 * regularity + nodes + 1);
    std::vector<unsigned int> renumbering(face_data.dofs_per_cell);
    hermite_hierarchic_to_lexicographic_numbering<dim - 1>(regularity,
                                                           nodes,
                                                           renumbering);
    Utilities::invert_permutation(renumbering);
    return renumbering;
  }

  template <int dim>
  TensorProductPolynomials<dim>
  get_hermite_polynomials(const unsigned int regularity)
  {
    TensorProductPolynomials<dim> poly_space(
      Polynomials::HermiteMaxreg::generate_complete_basis(regularity));
    std::vector<unsigned int> renumber =
      internal::hermite_face_lexicographic_to_hierarchic_numbering<dim + 1>(
        regularity, 0);
    poly_space.set_numbering(renumber);
    return poly_space;
  }
  
#if HERMITE_CUSTOM_FE_CLASS
  template <int dim>
  TensorProductPolynomials<dim>
  get_hermite_polynomials(const unsigned int regularity, const unsigned int nodes)
  {
      TensorProductPolynomials<dim> poly_space(
      Polynomials::HermiteCustomreg::generate_complete_basis(regularity, nodes));
    std::vector<unsigned int> renumber =
      internal::hermite_face_lexicographic_to_hierarchic_numbering<dim + 1>(
        regularity, nodes);
    poly_space.set_numbering(renumber);
    return poly_space;
  }
#endif

  inline unsigned int
  binomial(const unsigned int n, const unsigned int i)
  {
    unsigned int C = 1, k = 1;
    for (unsigned int j = n; j > i; j--)
      {
        C *= j;
        C /= k++;
      }
    return C;
  }
} // namespace internal


/*
 * Implementation structs for providing constraint matrices for hanging nodes
 */
template <int xdim, int xspacedim>
struct FE_Hermite<xdim, xspacedim>::Implementation
{
  static void
  create_F_matrix(std::vector<double> &F_matrix, const unsigned int regularity)
  {
    const unsigned int        sz = regularity + 1;
    std::vector<double>       bhalf_matrix(sz * sz);
    std::vector<unsigned int> bzero_inv_matrix(sz * sz);
    F_matrix.resize(sz * sz);
    double       big_factor = 1, local_factor;
    unsigned int min_ij, diag_value;
    int          sign_1 = 1, sign_2, sign_3, temp;
    for (unsigned int i = 0; i < sz; i++)
      {
        big_factor *= 2;
        bzero_inv_matrix[i + i * sz] = 1;
        sign_2                       = sign_1;
        for (unsigned int j = 0; j < i; j++)
          {
            sign_3 = sign_2;
            temp   = -sign_2 * internal::binomial(sz, i - j);
            for (unsigned int k = j + 1; k < i; k++)
              {
                sign_3 = -sign_3;
                temp -= sign_3 * internal::binomial(sz, i - k) *
                        bzero_inv_matrix[j + k * sz];
              }
            bzero_inv_matrix[j + i * sz] = temp;
            sign_2                       = -sign_2;
          }
        sign_1 = -sign_1;
      }
    sign_1     = 1;
    diag_value = 1;
    for (unsigned int i = 0; i < sz; i++)
      {
        local_factor = big_factor;
        for (unsigned int j = 0; j < sz; j++)
          {
            F_matrix[j + i * sz] = 0;
            bzero_inv_matrix[i + j * sz] *= diag_value;
            min_ij = (i < j) ? i : j;
            temp = 0, sign_2 = 1;
            for (unsigned int k = 0; k < min_ij; k++)
              {
                temp += sign_2 * internal::binomial(j, k) *
                        internal::binomial(sz, i - k);
                sign_2 = -sign_2;
              }
            bhalf_matrix[j + i * sz] = sign_1 * local_factor * temp;
            local_factor *= 2;
          }
        sign_1 = -sign_1;
        big_factor *= 0.5;
        diag_value *= 4;
      }
    for (unsigned int i = 0; i < sz; i++)
      for (unsigned int k = 0; k < sz; k++)
        for (unsigned int j = 0; j < sz; j++)
          F_matrix[j + i * sz] +=
            bhalf_matrix[k + i * sz] * bzero_inv_matrix[j + k * sz];
    big_factor = 1;
    for (unsigned int i = 0; i < sz; i++)
      {
        for (unsigned int j = 0; j < sz; j++)
          F_matrix[j + i * sz] *= big_factor;
        big_factor *= i + 1;
      }
    return;
  }
  
#if HERMITE_CUSTOM_FE_CLASS
  static void
  create_F_matrix(std::vector<double> &      F_matrix,
                  const unsigned int         regularity,
                  const unsigned int         nodes)
  {
      Assert(false, ExcNotImplemented());
  }
#endif

  static inline void
  create_G_matrix(std::vector<double> &      G_matrix,
                  const std::vector<double> &F_matrix,
                  const unsigned int         regularity)
  {
    const unsigned int sz = regularity + 1;
    Assert(F_matrix.size() == sz * sz,
           ExcMessage(
             "Provided F_matrix has the wrong size for the regularity."));
    G_matrix.resize(sz * sz);
    int          sign_1, sign_2;
    sign_1 = 1;
    for (unsigned int i = 0; i < sz; i++)
      {
        sign_2 = sign_1;
        for (unsigned int j = 0; j < sz; j++)
          {
            G_matrix[j + i * sz] = sign_2 * F_matrix[j + i * sz];
            sign_2               = -sign_2;
          }
        sign_1 = -sign_1;
      }
    return;
  }
  
#if HERMITE_CUSTOM_FE_CLASS
  static inline void
  create_G_matrix(std::vector<double> &      G_matrix,
                  const std::vector<double> &F_matrix,
                  const unsigned int         regularity,
                  const unsigned int         nodes)
  {
      Assert(false, ExcNotImplemented());
  }
#endif

  template <int spacedim>
  static void
  initialise_constraints(FE_Hermite<1, spacedim> &)
  {
    // Not needed for 1D
  }

  template <int spacedim>
  static void
  initialise_constraints(FE_Hermite<2, spacedim> &fe_hermite)
  {
    const unsigned int nodes = fe_hermite.nodes;
    if (nodes == 0)
    {    
        const unsigned int  regularity = fe_hermite.regularity;
        const unsigned int  sz         = regularity + 1;
        std::vector<double> F_matrix(sz * sz), G_matrix(sz * sz);

        create_F_matrix(F_matrix, regularity);
        create_G_matrix(G_matrix, F_matrix, regularity);
        std::vector<unsigned int> face_index_map =
            internal::hermite_face_lexicographic_to_hierarchic_numbering<2>(
            regularity, 0);
        fe_hermite.interface_constraints.TableBase<2, double>::reinit(
        fe_hermite.interface_constraints_size());
        for (unsigned int i = 0; i < sz; i++)
            for (unsigned int j = 0; j < 2 * sz; j++)
                fe_hermite.interface_constraints(i, face_index_map[j]) =
                  static_cast<double>(i == j);
        for (unsigned int i = 0; i < sz; i++)
            for (unsigned int j = 0; j < sz; j++)
            {
                fe_hermite.interface_constraints(sz + i, face_index_map[j]) =
                  F_matrix[j + i * sz];
                fe_hermite.interface_constraints(sz + i, face_index_map[sz + j]) =
                  G_matrix[j + i * sz];
            }
        for (unsigned int i = 0; i < sz; i++)
            for (unsigned int j = 0; j < sz; j++)
            {
                fe_hermite.interface_constraints(2 * sz + i, face_index_map[j]) =
                  F_matrix[j + i * sz];
                fe_hermite.interface_constraints(2 * sz + i, face_index_map[sz + j]) =
                  G_matrix[j + i * sz];
            }
        for (unsigned int i = 0; i < sz; i++)
            for (unsigned int j = 0; j < 2 * sz; j++)
                fe_hermite.interface_constraints(3 * sz + i, face_index_map[j]) =
                  (i + sz == j) ? 1 : 0;
    }
    else
    {
        Assert(false, ExcNotImplemented());
    }
    return;
  }

  template <int spacedim>
  static void
  initialise_constraints(FE_MaxHermite<3, spacedim> &fe_hermite)
  {
      const unsigned int nodes = fe_hermite.nodes;
      if (nodes == 0)
      {
        const unsigned int  regularity = fe_hermite.regularity;
        const unsigned int  sz         = regularity + 1;
        const unsigned int  sz2        = sz * sz;
        std::vector<double> F_matrix(sz * sz), G_matrix(sz * sz);
        create_F_matrix(F_matrix, regularity);
        create_G_matrix(G_matrix, F_matrix, regularity);
        std::vector<unsigned int> face_index_map =
          internal::hermite_face_lexicographic_to_hierarchic_numbering<3>(
          regularity, 0);
        fe_hermite.interface_constraints.TableBase<2, double>::reinit(
          fe_hermite.interface_constraints_size());
        unsigned int local_1, local_2;
        double       entry_1, entry_2, entry_3, entry_4, temp;
        for (unsigned int i = 0; i < sz2; i++)
            for (unsigned int j = 0; j < 4 * sz * sz; j++)
            {
                entry_1 = static_cast<double>(i == j);
                fe_hermite.interface_constraints(i, face_index_map[j])    = entry_1;
                fe_hermite.interface_constraints(i + 3 * sz2,
                                           face_index_map[j + sz2]) = entry_1;
                fe_hermite.interface_constraints(i + 12 * sz2,
                                           face_index_map[j + 2 * sz2]) =
                  entry_1;
                fe_hermite.interface_constraints(i + 15 * sz2,
                                           face_index_map[j + 3 * sz2]) =
                  entry_1;
            }
        for (unsigned int i = 0; i < sz; i++)
            for (unsigned int j = 0; j < sz; j++)
            {
                entry_1 = F_matrix[j + i * sz];
                entry_2 = G_matrix[j + i * sz];
                for (unsigned int k = 0; k < sz; k++)
                    for (unsigned int l = 0; l < sz; l++)
                    {
                        entry_4 = static_cast<double>(k == l);
                        entry_3 = entry_1 * entry_4;
                        entry_4 *= entry_2;
                        local_1 = k + i * sz + 4 * sz2;
                        local_2 = l + j * sz;
                        fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                          entry_3;
                        fe_hermite.interface_constraints(
                          local_1, face_index_map[local_2 + 2 * sz2]) = entry_4;
                        local_1 += 3 * sz2;
                        local_2 += sz2;
                        fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                          entry_3;
                        fe_hermite.interface_constraints(
                          local_1, face_index_map[local_2 + 2 * sz2]) = entry_4;
                        local_1 += sz2;
                        local_2 -= sz2;
                        fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                          entry_3;
                        fe_hermite.interface_constraints(
                          local_1, face_index_map[local_2 + 2 * sz2]) = entry_4;
                        local_1 += 3 * sz2;
                        local_2 += sz2;
                        fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                          entry_3;
                        fe_hermite.interface_constraints(
                        local_1, face_index_map[local_2 + 2 * sz2]) = entry_4;
                    }
            entry_1 = static_cast<double>(i == j);
            for (unsigned int k = 0; k < sz; k++)
                for (unsigned int l = 0; l < sz; l++)
                {
                    entry_3 = entry_1 * F_matrix[l + k * sz];
                    entry_4 = entry_1 * G_matrix[l + k * sz];
                    local_1 = k * i * sz + sz2;
                    local_2 = l + j * sz;
                    fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                        entry_3;
                    fe_hermite.interface_constraints(
                        local_1, face_index_map[local_2 + sz2]) = entry_4;
                    local_1 += sz2;
                    fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                      entry_3;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + sz2]) = entry_4;
                    local_1 += 11 * sz2;
                    local_2 += 2 * sz2;
                    fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                      entry_3;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + sz2]) = entry_4;
                    local_1 += sz2;
                    fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                      entry_3;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + sz2]) = entry_4;
                }
            entry_1 = F_matrix[j + i * sz];
            entry_3 = G_matrix[j + i * sz];
            for (unsigned int k = 0; k < sz; k++)
                for (unsigned int l = 0; l < sz; l++)
                {
                    temp    = G_matrix[l + k * sz];
                    entry_2 = entry_1 * temp;
                    entry_4 = entry_3 * temp;
                    temp    = F_matrix[l + k * sz];
                    entry_1 *= temp;
                    entry_3 *= temp;
                    local_1 = k + i * sz + 5 * sz2;
                    local_2 = l + j * sz;
                    fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                      entry_1;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + sz2]) = entry_2;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + 2 * sz2]) = entry_3;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + 3 * sz2]) = entry_4;
                    local_1 += sz2;
                    fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                      entry_1;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + sz2]) = entry_2;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + 2 * sz2]) = entry_3;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + 3 * sz2]) = entry_4;
                    local_1 += 3 * sz2;
                    fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                      entry_1;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + sz2]) = entry_2;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + 2 * sz2]) = entry_3;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + 3 * sz2]) = entry_4;
                    local_1 += sz2;
                    fe_hermite.interface_constraints(local_1,
                                                 face_index_map[local_2]) =
                      entry_1;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + sz2]) = entry_2;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + 2 * sz2]) = entry_3;
                    fe_hermite.interface_constraints(
                      local_1, face_index_map[local_2 + 3 * sz2]) = entry_4;
                }
            }
        }
        else
        {
            Assert(false, ExcNotImplemented());
        }
    return;
  }
};


/*
 * Member functions for the Hermite class
 */
//Constructors
template <int dim, int spacedim>
FE_Hermite<dim, spacedim>::FE_Hermite(const unsigned int reg)
  : FE_Poly<dim, spacedim>(
      internal::get_hermite_polynomials<dim>(reg),
      FiniteElementData<dim>(internal::get_hermite_dpo_vector(dim, reg, 0),
                             1,
                             2 * reg + 1,
                             (reg ? FiniteElementData<dim>::H2 :
                                    FiniteElementData<dim>::H1)),
      std::vector<bool>(Utilities::pow(2 * (reg + 1), dim), false),
      std::vector<ComponentMask>(Utilities::pow(2 * (reg + 1), dim),
                                 std::vector<bool>(1, true)))
  , regularity(reg), nodes(0)
{
    std::vector<unsigned int> renumber =
    internal::hermite_face_lexicographic_to_hierarchic_numbering<dim + 1>(
      regularity, 0);
    this->poly_space.set_numbering(renumber);
}

#if HERMITE_CUSTOM_FE_CLASS
template <int dim, int spacedim>
FE_Hermite<dim, spacedim>::FE_Hermite(const unsigned int reg, const unsigned int nodes)
  : FE_Poly<dim, spacedim>(
      internal::get_hermite_polynomials<dim>(reg, nodes),
      FiniteElementData<dim>(internal::get_hermite_dpo_vector(dim, reg, nodes),
                             1,
                             nodes + 2 * reg + 1,
                             (reg ? FiniteElementData<dim>::H2 :
                                    FiniteElementData<dim>::H1)),
      std::vector<bool>(Utilities::pow(nodes + 2 * reg + 2, dim), false),
      std::vector<ComponentMask>(Utilities::pow(nodes + 2 * reg + 2, dim),
                                 std::vector<bool>(1, true)))
  , regularity(reg), nodes(nodes)
{
    std::vector<unsigned int> renumber =
    internal::hermite_face_lexicographic_to_hierarchic_numbering<dim + 1>(
      regularity, nodes);
    this->poly_space.set_numbering(renumber);
}
#endif

template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::initialize_constraints()
{
  Implementation::initialise_constraints(*this);
  return;
}

template <int dim, int spacedim>
std::string
FE_Hermite<dim, spacedim>::get_name() const
{
  std::ostringstream name_buffer;
  name_buffer << "FE_Hermite<" << dim << "," << spacedim << ">("
              << this->regularity << "," << this->nodes ")";
  return name_buffer.str();
}

template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_Hermite<dim, spacedim>::clone() const
{
  return std::make_unique<FE_Hermite<dim, spacedim>>(*this);
}
/*
template <int dim, int spacedim>
void FE_Hermite<dim, spacedim>::get_interpolation_matrix(const
FiniteElement<dim, spacedim> &source, FullMatrix<double> &matrix) const
{
    const unsigned int dofs = this->dofs_per_cell;
    const unsigned int source_dofs = source.dofs_per_cell;
    Assert( matrix.m() == dofs, ExcDimensionMismatch(interpolation_matrix.m(),
dofs)); Assert( matrix.n() == source_dofs,
ExcDimensionMismatch(interpolation_matrix.n(), source_dofs));
        // Evaluation must be performed by projection

}
*/

#include "fe_hermite.inst.in"

DEAL_II_NAMESPACE_CLOSE
