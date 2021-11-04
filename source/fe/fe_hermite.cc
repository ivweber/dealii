/**
 * Source code for Hermite basis functions.
 */

#include <deal.II/base/template_constraints.h>
#include <deal.II/base/table.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_hermite.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/matrix_tools.h>

#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <iostream>
#include <algorithm>

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
  
  /**
   * Renumbering function. Function needs different levels of for loop nesting
   * for different values of dim, so different definitions are used for 
   * simplicity.
   */
  template <int dim>
  void
  hermite_hierarchic_to_lexicographic_numbering(const unsigned int regularity,
                                                const unsigned int nodes,
                                                std::vector<unsigned int> &h2l);
  
  template<>
  void
  hermite_hierarchic_to_lexicographic_numbering<1>(const unsigned int regularity,
                                                const unsigned int nodes,
                                                std::vector<unsigned int> &h2l)
  {
        const unsigned int node_dofs_1d = regularity + 1;
        
        AssertDimension(h2l.size(), 2 * node_dofs_1d + nodes);
        
        unsigned int count = 0;
        // Assign DOFs at vertices
        for (unsigned int di = 0; di < 2; ++di)
            for (unsigned int i = 0; i < node_dofs_1d; ++i, ++count)
                h2l[i + di * node_dofs_1d] = i + di * (node_dofs_1d + nodes);
            
        // Assign DOFs on line if needed
        for (unsigned int i = 0; i < nodes; ++i, ++count)
            h2l[i + 2 * node_dofs_1d] = i + node_dofs_1d;
        
        AssertDimension(count, 2 * node_dofs_1d + nodes);
  }
  
  template<>
  void
  hermite_hierarchic_to_lexicographic_numbering<2>(const unsigned int regularity,
                                                const unsigned int nodes,
                                                std::vector<unsigned int> &h2l)
  {
        const unsigned int node_dofs_1d = regularity + 1;
        
        const unsigned int dim_dofs_1d  = 2 * node_dofs_1d + nodes;
        
        AssertDimension(h2l.size(), dim_dofs_1d * dim_dofs_1d);
        
        unsigned int count = 0, offset = 0;
        
        // Assign DOFs at vertices
        for (unsigned int di = 0; di < 2; ++di)
            for (unsigned int dj = 0; dj < 2; ++dj)
            {
                for (unsigned int i = 0; i < node_dofs_1d; ++i)
                    for (unsigned int j = 0; j < node_dofs_1d; ++j, ++count)
                        h2l[j + i * node_dofs_1d + offset] = 
                            j + i * dim_dofs_1d + (dj + di * dim_dofs_1d) * (node_dofs_1d + nodes);
                offset += node_dofs_1d * node_dofs_1d;
            }
                
        if (nodes)
        {
            // Assign DOFs on edges
            for (unsigned int i = 0; i < nodes; ++i)
                for (unsigned int dj = 0; dj < 2; ++dj)
                    for (unsigned int j = 0; j < node_dofs_1d; ++j, ++count)
                        h2l[j + (2 * i + dj) * node_dofs_1d + offset] =
                            j + (i + node_dofs_1d) * dim_dofs_1d + dj * (node_dofs_1d +
                            nodes);
            offset += 2 * nodes * node_dofs_1d;
            
            for (unsigned int i = 0; i < nodes; ++i)
                for (unsigned int di = 0; di < 2; ++di)
                    for (unsigned int j = 0; j < node_dofs_1d; ++j, ++count)
                        h2l[j + (2 * i + di) * node_dofs_1d + offset] =
                            i + j * dim_dofs_1d + di * (node_dofs_1d + nodes) * dim_dofs_1d +
                            node_dofs_1d;
            offset += 2 * nodes * node_dofs_1d;
            
            // Assign DOFs on face
            for (unsigned int i = 0; i < nodes; ++i)
                for (unsigned int j = 0; j < nodes; ++j, ++count)
                    h2l[j + i * nodes + offset] =
                        j + (i + node_dofs_1d) * dim_dofs_1d + node_dofs_1d;
        }
            
        AssertDimension(count, dim_dofs_1d * dim_dofs_1d);
  }  
  
  template<>
  void
  hermite_hierarchic_to_lexicographic_numbering<3>(const unsigned int regularity,
                                                const unsigned int nodes,
                                                std::vector<unsigned int> &h2l)
  {
        const unsigned int node_dofs_1d = regularity + 1;
        const unsigned int node_dofs_2d = node_dofs_1d * node_dofs_1d;
    
        const unsigned int dim_dofs_1d  = 2 * node_dofs_1d + nodes;
        const unsigned int dim_dofs_2d  = dim_dofs_1d * dim_dofs_1d;
    
        AssertDimension(h2l.size(), dim_dofs_2d * dim_dofs_1d);
    
        unsigned int offset = 0, count = 0;
    
        // Assign DOFs at nodes
        for (unsigned int di = 0; di < 2; ++di)
            for (unsigned int dj = 0; dj < 2; ++dj)
                for (unsigned int dk = 0; dk < 2; ++dk)
                {
                    for (unsigned int i = 0; i < node_dofs_1d; ++i)
                        for (unsigned int j = 0; j < node_dofs_1d; ++j)
                            for (unsigned int k = 0; k < node_dofs_1d; ++k, ++count)
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
            for (unsigned int i = 0; i < nodes; ++i)
                for (unsigned int dj = 0; dj < 2; ++dj)
                    for (unsigned int dk = 0; dk < 2; ++dk)
                    {
                        for (unsigned int j = 0; j < node_dofs_1d; ++j)
                            for (unsigned int k = 0; k < node_dofs_1d; ++k, ++count)
                                h2l[k + j * node_dofs_1d + offset] =
                                    k + j * dim_dofs_1d +
                                    (i + node_dofs_1d) * dim_dofs_2d +
                                    (node_dofs_1d + nodes) * (dk + dj * dim_dofs_1d);
                        offset += node_dofs_2d;
                    }
                    
            // edges parallel to y
            for (unsigned int j = 0; j < nodes; ++j)
                for (unsigned int di = 0; di < 2; ++di)
                    for (unsigned int dk = 0; dk < 2; ++dk)
                    {
                        for (unsigned int i = 0; i < node_dofs_1d; ++i)
                            for (unsigned int k = 0; k < node_dofs_1d; ++k, ++count)
                                h2l[k + i * node_dofs_1d + offset] =
                                    k + i * dim_dofs_2d +
                                    (j + node_dofs_1d) * dim_dofs_1d +
                                    (node_dofs_1d + nodes) * (dk + di * dim_dofs_2d);
                        offset += node_dofs_2d;
                    }
                    
            // edges parallel to x
            for (unsigned int k = 0; k < nodes; ++k)
                for (unsigned int di = 0; di < 2; ++di)
                    for (unsigned int dj = 0; dj < 2; ++dj)
                    {
                        for (unsigned int i = 0; i < node_dofs_1d; ++i)
                            for (unsigned int j = 0; j < node_dofs_1d; ++j, ++count)
                                h2l[j + i * node_dofs_1d + offset] =
                                    j * dim_dofs_1d + i * dim_dofs_2d + k +
                                    node_dofs_1d +
                                    (node_dofs_1d + nodes) *
                                    (dj * dim_dofs_1d + di * dim_dofs_2d);
                        offset += node_dofs_2d;
                    }
                    
            // Assign DOFs on faces
            // faces normal to x
            for (unsigned int i = 0; i < nodes; ++i)
                for (unsigned int j = 0; j < nodes; ++j)
                    for (unsigned int dk = 0; dk < 2; ++dk)
                    {
                        for (unsigned int k = 0; k < node_dofs_1d; ++k, ++count)
                            h2l[k + offset] = k + (j + node_dofs_1d) * dim_dofs_1d +
                                (i + node_dofs_1d) * dim_dofs_2d +
                                (node_dofs_1d + nodes) * dk;
                        offset += node_dofs_1d;
                    }
                    
            // faces normal to y
            for (unsigned int i = 0; i < nodes; ++i)
                for (unsigned int k = 0; k < nodes; ++k)
                    for (unsigned int dj = 0; dj < 2; ++dj)
                    {
                        for (unsigned int j = 0; j < node_dofs_1d; ++j, ++count)
                            h2l[j + offset] =
                                j * dim_dofs_1d + k + node_dofs_1d +
                                (i + node_dofs_1d) * dim_dofs_2d +
                                (node_dofs_1d + nodes) * dj * dim_dofs_1d;
                        offset += node_dofs_1d;
                    }
                    
            // faces normal to z
            for (unsigned int j = 0; j < nodes; ++j)
                for (unsigned int k = 0; k < nodes; ++k)
                    for (unsigned int di = 0; di < 2; ++di)
                    {
                        for (unsigned int i = 0; i < node_dofs_1d; ++i, ++count)
                            h2l[i + offset] =
                                i * dim_dofs_2d + k + node_dofs_1d +
                                (j + node_dofs_1d) * dim_dofs_1d +
                                (node_dofs_1d + nodes) * di * dim_dofs_2d;
                        offset += node_dofs_1d;
                    }
                    
            // Assign DOFs in cell
            for (unsigned int i = 0; i < nodes; ++i)
                for (unsigned int j = 0; j < nodes; ++j)
                    for (unsigned int k = 0; k < nodes; ++k, ++count)
                        h2l[k + (j + i * nodes) * nodes + offset] =
                            k + node_dofs_1d + (j + node_dofs_1d) * dim_dofs_1d +
                            (i + node_dofs_1d) * dim_dofs_2d;
        }
        
        AssertDimension(count, dim_dofs_1d * dim_dofs_2d);
  }
          
          
  template <>
  void
  hermite_hierarchic_to_lexicographic_numbering<4>(const unsigned int regularity,
                                                   const unsigned int nodes,
                                                   std::vector<unsigned int> &h2l)
  {
        const unsigned int node_dofs_1d = regularity + 1;
        const unsigned int node_dofs_2d = node_dofs_1d * node_dofs_1d;
        const unsigned int node_dofs_3d = node_dofs_2d * node_dofs_1d;
      
        const unsigned int dim_dofs_1d  = 2 * node_dofs_1d + nodes;
        const unsigned int dim_dofs_2d  = dim_dofs_1d * dim_dofs_1d;
        const unsigned int dim_dofs_3d  = dim_dofs_2d * dim_dofs_1d;
    
        unsigned int       offset = 0, count = 0;
    
        AssertDimension(h2l.size(), dim_dofs_2d * dim_dofs_2d);

        // Assign DOFs at nodes
        for (unsigned int di = 0; di < 2; ++di)
            for (unsigned int dj = 0; dj < 2; ++dj)
                for (unsigned int dk = 0; dk < 2; ++dk)
                    for (unsigned int dl = 0; dl < 2; ++dl)
                    {
                        for (unsigned int i = 0; i < node_dofs_1d; ++i)
                            for (unsigned int j = 0; j < node_dofs_1d; ++j)
                                for (unsigned int k = 0; k < node_dofs_1d; ++k)
                                    for (unsigned int l = 0; l < node_dofs_1d; ++l, ++count)
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

        AssertDimension(count, dim_dofs_2d * dim_dofs_2d);
  }

  template <int dim>
  inline std::vector<unsigned int>
  hermite_hierarchic_to_lexicographic_numbering(
    const unsigned int regularity,
    const unsigned int nodes)
  {
    const std::vector<unsigned int> dpo =
      get_hermite_dpo_vector(dim, regularity, nodes);
    const dealii::FiniteElementData<dim> face_data(
      dpo, 1, 2 * regularity + nodes + 1);
    std::vector<unsigned int> renumbering(face_data.dofs_per_cell);
    hermite_hierarchic_to_lexicographic_numbering<dim>(regularity,
                                                       nodes,
                                                       renumbering);
    return renumbering;
  }
    
  template <int dim>
  inline std::vector<unsigned int>
  hermite_lexicographic_to_hierarchic_numbering(
    const unsigned int regularity,
    const unsigned int nodes)
  {    
    return Utilities::invert_permutation(hermite_hierarchic_to_lexicographic_numbering<dim>(regularity,nodes));
  }
  
  template <int dim>
  inline std::vector<unsigned int>
  hermite_face_lexicographic_to_hierarchic_numbering(
    const unsigned int regularity,
    const unsigned int nodes)
  {
    if (dim <= 1)
      return std::vector<unsigned int>();
    else
      return hermite_lexicographic_to_hierarchic_numbering<dim - 1>(regularity, nodes);
  }

  template <int dim>
  TensorProductPolynomials<dim>
  get_hermite_polynomials(const unsigned int regularity)
  {
    TensorProductPolynomials<dim> poly_space(
      Polynomials::HermiteMaxreg::generate_complete_basis(regularity));
    std::vector<unsigned int> renumber =
      internal::hermite_hierarchic_to_lexicographic_numbering<dim>(
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
    for (unsigned int j = n; j > i; --j)
      {
        C *= j;
        C /= k++;
      }
    return C;
  }
  
  template <int xdim, int xspacedim = xdim, typename xNumber = double>
  class Rescaler
  {
  public:
      void
      rescale_fe_hermite_values(//Rescaler<dim, spacedim, Number> &                           rescaler,
                                const FE_Hermite<xdim, xspacedim> &                           fe_herm,
                                const typename MappingHermite<xdim, xspacedim>::InternalData &mapping_data,
                                Table<2, xNumber> &                                          value_list);
  
      //TODO: Adapt these functions to work with DOFs assigned on edges, faces etc
      template <int spacedim, typename Number>
      void 
      rescale_fe_hermite_values(Rescaler<1, spacedim, Number> &                           /*rescaler*/,
                                const FE_Hermite<1, spacedim> &                           fe_herm,
                                const typename MappingHermite<1, spacedim>::InternalData &mapping_data,
                                Table<2, Number> &                                        value_list)
      {
          const unsigned int dofs = fe_herm.n_dofs_per_cell();
          (void)dofs;
          
          AssertDimension(value_list.size(0), dofs);
          
          const unsigned int regularity = fe_herm.get_regularity();
          const unsigned int nodes = 0;//fe_herm.n_nodes();
          
          AssertDimension(dofs, 2 * regularity + nodes + 2);
          
          const unsigned int q_points = value_list.size(1);
          AssertIndexRange(q_points, mapping_data.quadrature_points.size() + 1);
        
          std::vector<unsigned int> l2h = hermite_lexicographic_to_hierarchic_numbering<1>(regularity, nodes);
          
          for (unsigned int q = 0; q < q_points; ++q)
          {
              double factor_1 = 1.0;
            
              for (unsigned int d1 = 0, d2 = regularity + nodes + 1; d2 < dofs; ++d1, ++d2)
              {
                  value_list(l2h[d1], q) *= factor_1;
                  value_list(l2h[d2], q) *= factor_1; // This appears to be rescaled less than the above entries, by a factor of about factor_1
                                                                 //  
                  factor_1 *= mapping_data.cell_extents[0];     //   This seems to be reading the correct value for the cells, and calculating factor_1 correctly    
              }
          }
      }
      
      template <int spacedim, typename Number>
      void 
      rescale_fe_hermite_values(Rescaler<2, spacedim, Number> &                           /*rescaler*/,
                                const FE_Hermite<2, spacedim> &                           fe_herm,
                                const typename MappingHermite<2, spacedim>::InternalData &mapping_data,
                                Table<2, Number> &                                        value_list)
      {
          const unsigned int dofs = fe_herm.n_dofs_per_cell();
          (void)dofs;
          
          AssertDimension(value_list.size(0), dofs);
          
          const unsigned int regularity = fe_herm.get_regularity();
          const unsigned int nodes = 0;//fe_herm.n_nodes();
          const unsigned int dofs_per_dim = 2 * regularity + nodes + 2;
          
          AssertDimension(dofs_per_dim * dofs_per_dim, dofs);
          
          const unsigned int q_points = value_list.size(1);
          AssertIndexRange(q_points, mapping_data.quadrature_points.size() + 1);
          
          std::vector<unsigned int> l2h = hermite_lexicographic_to_hierarchic_numbering<2>(regularity, nodes);
      
          AssertDimension(l2h.size(), dofs);
          
          //Something is going wrong below. The rescaling factors appear to be correct, but dofs running in
          //the x-direction are not rescaled properly, causing a factor 2 jump going from x- to x+ at each 
          //cell boundary
          FullMatrix<double> factors(dofs_per_dim, dofs_per_dim);
          for (unsigned int q = 0; q < q_points; ++q)
          {
              //double factor_2 = 1.0;
                    
              for (unsigned int d3 = 0, d4 = regularity + nodes + 1; d4 < dofs_per_dim; ++d3, ++d4)
              {
                  //double factor_1 = factor_2;
                
                  for (unsigned int d1 = 0, d2 = regularity + nodes + 1; d2 < dofs_per_dim; ++d1, ++d2)
                  {
                      double factor_1 = std::pow(mapping_data.cell_extents[0], d1) * std::pow(mapping_data.cell_extents[1], d3);
                      factors(d1,d3) = factor_1;
                      factors(d2,d3) = factor_1;
                      factors(d1,d4) = factor_1;
                      factors(d2,d4) = factor_1;
                      
                      value_list(l2h[d1 + d3 * dofs_per_dim], q) *= factor_1;
                      value_list(l2h[d2 + d3 * dofs_per_dim], q) *= factor_1;
                      value_list(l2h[d1 + d4 * dofs_per_dim], q) *= factor_1;
                      value_list(l2h[d2 + d4 * dofs_per_dim], q) *= factor_1;
                            
                      //factor_1 *= mapping_data.cell_extents[0];
                  }
                
                  //factor_2 *= mapping_data.cell_extents[1];
              }
          }
      }
  
      template <int spacedim, typename Number>
      void
      rescale_fe_hermite_values(Rescaler<3, spacedim, Number> &                           /*rescaler*/,
                                const FE_Hermite<3, spacedim> &                           fe_herm,
                                const typename MappingHermite<3, spacedim>::InternalData &mapping_data,
                                Table<2, Number> &                                        value_list)
      {
          const unsigned int dofs = fe_herm.n_dofs_per_cell();
          (void)dofs;
          
          AssertDimension(value_list.size(0), dofs);
          
          const unsigned int regularity = fe_herm.get_regularity();
          const unsigned int nodes = 0;//fe_herm.n_nodes();
          const unsigned int dofs_per_dim = 2 * regularity + nodes + 2;
          
          AssertDimension(dofs_per_dim * dofs_per_dim * dofs_per_dim, dofs);
      
          const unsigned int q_points = value_list.size(1);
          AssertIndexRange(q_points, mapping_data.quadrature_points.size() + 1);
          
          std::vector<unsigned int> l2h = hermite_lexicographic_to_hierarchic_numbering<3>(regularity, nodes);
          
          for (unsigned int q = 0; q < q_points; ++q)
          {
              double factor_3 = 1.0;
                    
              for (unsigned int d5 = 0, d6 = regularity + nodes + 1; d6 < dofs_per_dim; ++d5, ++d6)
              {
                  double factor_2 = factor_3;

                  for (unsigned int d3 = 0, d4 = regularity + nodes + 1; d4 < dofs_per_dim; ++d3, ++d4)
                  {
                      double factor_1 = factor_2;
                        
                      for (unsigned int d1 = 0, d2 = regularity + nodes + 1; d2 < dofs_per_dim; ++d1, ++d2)
                      {
                          value_list(l2h[d1 + d3 * dofs_per_dim + d5 * dofs_per_dim * dofs_per_dim], q) 
                            *= factor_1;
                          value_list(l2h[d2 + d3 * dofs_per_dim + d5 * dofs_per_dim * dofs_per_dim], q) 
                            *= factor_1;
                          value_list(l2h[d1 + d4 * dofs_per_dim + d5 * dofs_per_dim * dofs_per_dim], q) 
                            *= factor_1;
                          value_list(l2h[d2 + d4 * dofs_per_dim + d5 * dofs_per_dim * dofs_per_dim], q) 
                            *= factor_1;
                          value_list(l2h[d1 + d3 * dofs_per_dim + d6 * dofs_per_dim * dofs_per_dim], q) 
                            *= factor_1;
                          value_list(l2h[d2 + d3 * dofs_per_dim + d6 * dofs_per_dim * dofs_per_dim], q) 
                            *= factor_1;
                          value_list(l2h[d1 + d4 * dofs_per_dim + d6 * dofs_per_dim * dofs_per_dim], q) 
                            *= factor_1;
                          value_list(l2h[d2 + d4 * dofs_per_dim + d6 * dofs_per_dim * dofs_per_dim], q) 
                            *= factor_1;
                            
                          factor_1 *= mapping_data.cell_extents[0];
                      }
                    
                      factor_2 *= mapping_data.cell_extents[1];
                  }
                
                  factor_3 *= mapping_data.cell_extents[2];
              }
          }
      }
  };
}   //namespace internal
  

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
    for (unsigned int i = 0; i < sz; ++i)
      {
        big_factor *= 2;
        bzero_inv_matrix[i + i * sz] = 1;
        sign_2                       = sign_1;
        for (unsigned int j = 0; j < i; ++j)
          {
            sign_3 = sign_2;
            temp   = -sign_2 * internal::binomial(sz, i - j);
            for (unsigned int k = j + 1; k < i; ++k)
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
    for (unsigned int i = 0; i < sz; ++i)
      {
        local_factor = big_factor;
        for (unsigned int j = 0; j < sz; ++j)
          {
            F_matrix[j + i * sz] = 0;
            bzero_inv_matrix[i + j * sz] *= diag_value;
            min_ij = (i < j) ? i : j;
            temp = 0, sign_2 = 1;
            for (unsigned int k = 0; k < min_ij; ++k)
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
    for (unsigned int i = 0; i < sz; ++i)
      for (unsigned int k = 0; k < sz; ++k)
        for (unsigned int j = 0; j < sz; ++j)
          F_matrix[j + i * sz] +=
            bhalf_matrix[k + i * sz] * bzero_inv_matrix[j + k * sz];
    big_factor = 1;
    for (unsigned int i = 0; i < sz; ++i)
      {
        for (unsigned int j = 0; j < sz; ++j)
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
    for (unsigned int i = 0; i < sz; ++i)
      {
        sign_2 = sign_1;
        for (unsigned int j = 0; j < sz; ++j)
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
  initialise_constraints(FE_Hermite<1, spacedim> &fe_hermite)
  {
    const unsigned int nodes = fe_hermite.nodes;
    if (nodes == 0)
    {
        const unsigned int regularity = fe_hermite.regularity;
        const unsigned int sz = regularity + 1;
        
        fe_hermite.interface_constraints.TableBase<2,double>::reinit(
            fe_hermite.interface_constraints_size());
        
        for (unsigned int i = 0; i < sz; ++i)
            fe_hermite.interface_constraints(i,i) =
                std::pow(0.5, i);
        //implement matching of function coefficients based on cell size here
    }
    else
    {
        Assert(false, ExcNotImplemented());
    }
  }

  template <int spacedim>
  static void
  initialise_constraints(FE_Hermite<2, spacedim> &fe_hermite)   //This function probably needs to be rewritten
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
        for (unsigned int i = 0; i < sz; ++i)
            for (unsigned int j = 0; j < 2 * sz; ++j)
                fe_hermite.interface_constraints(i, face_index_map[j]) =
                  static_cast<double>(i == j);
        for (unsigned int i = 0; i < sz; ++i)
            for (unsigned int j = 0; j < sz; ++j)
            {
                fe_hermite.interface_constraints(sz + i, face_index_map[j]) =
                  F_matrix[j + i * sz];
                fe_hermite.interface_constraints(sz + i, face_index_map[sz + j]) =
                  G_matrix[j + i * sz];
            }
        for (unsigned int i = 0; i < sz; ++i)
            for (unsigned int j = 0; j < sz; ++j)
            {
                fe_hermite.interface_constraints(2 * sz + i, face_index_map[j]) =
                  F_matrix[j + i * sz];
                fe_hermite.interface_constraints(2 * sz + i, face_index_map[sz + j]) =
                  G_matrix[j + i * sz];
            }
        for (unsigned int i = 0; i < sz; ++i)
            for (unsigned int j = 0; j < 2 * sz; ++j)
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
  initialise_constraints(FE_Hermite<3, spacedim> &fe_hermite)
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
        for (unsigned int i = 0; i < sz2; ++i)
            for (unsigned int j = 0; j < 4 * sz * sz; ++j)
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
        for (unsigned int i = 0; i < sz; ++i)
            for (unsigned int j = 0; j < sz; ++j)
            {
                entry_1 = F_matrix[j + i * sz];
                entry_2 = G_matrix[j + i * sz];
                for (unsigned int k = 0; k < sz; ++k)
                    for (unsigned int l = 0; l < sz; ++l)
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
            for (unsigned int k = 0; k < sz; ++k)
                for (unsigned int l = 0; l < sz; ++l)
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
            for (unsigned int k = 0; k < sz; ++k)
                for (unsigned int l = 0; l < sz; ++l)
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
   /* std::vector<unsigned int> renumber =
    internal::hermite_face_lexicographic_to_hierarchic_numbering<dim + 1>(
      regularity, 0);
    this->poly_space.set_numbering(renumber);*/
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
   /* std::vector<unsigned int> renumber =
    internal::hermite_face_lexicographic_to_hierarchic_numbering<dim + 1>(
      regularity, nodes);
    this->poly_space.set_numbering(renumber);*/
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
              << this->regularity << "," << this->nodes << ")";
  return name_buffer.str();
}

template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_Hermite<dim, spacedim>::clone() const
{
  return std::make_unique<FE_Hermite<dim, spacedim>>(*this);
}


template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::fill_fe_values(
  const typename Triangulation<dim, spacedim>::cell_iterator &,
  const CellSimilarity::Similarity                         cell_similarity,
  const Quadrature<dim> &                                  /*quadrature*/,
  const Mapping<dim, spacedim> &                           mapping,
  const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
  const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                     spacedim>
    &                                                            /*mapping_data*/,
  const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
  dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                     spacedim>
    &output_data) const
{
  // convert data object to internal
  // data for this class. fails with
  // an exception if that is not
  // possible
  Assert((dynamic_cast<const typename FE_Hermite<dim,spacedim>::InternalData *>(&fe_internal) != nullptr),
         ExcInternalError());
  const typename FE_Hermite<dim,spacedim>::InternalData &fe_data = static_cast<const typename FE_Hermite<dim,spacedim>::InternalData &>(fe_internal);
  
  Assert((dynamic_cast<const typename MappingHermite<dim, spacedim>::InternalData *>(&mapping_internal) != nullptr),
         ExcInternalError());
  const typename MappingHermite<dim, spacedim>::InternalData &mapping_internal_herm = static_cast<const typename MappingHermite<dim, spacedim>::InternalData &>(mapping_internal);

  const UpdateFlags flags(fe_data.update_each);

  // transform values gradients and higher derivatives. Values need to
  // be rescaled according the the nodal derivative they correspond to
  if ((flags & update_values) &&
      (cell_similarity != CellSimilarity::translation))
  {
      internal::Rescaler<dim, spacedim, double> shape_fix;
      for (unsigned int i=0; i<output_data.shape_values.size(0); ++i)
        for (unsigned int q=0; q<output_data.shape_values.size(1); ++q)
          output_data.shape_values(i, q) = fe_data.shape_values(i, q);
      shape_fix.rescale_fe_hermite_values(shape_fix,
                                          *this, 
                                          mapping_internal_herm, 
                                          output_data.shape_values);
  }
  
  if ((flags & update_gradients) &&
      (cell_similarity != CellSimilarity::translation))
  {
    for (unsigned int k = 0; k < this->n_dofs_per_cell(); ++k)
      mapping.transform(make_array_view(fe_data.shape_gradients, k),
                        mapping_covariant,
                        mapping_internal,
                        make_array_view(output_data.shape_gradients, k));
    
    internal::Rescaler<dim, spacedim, Tensor<1,spacedim>> grad_fix;
    grad_fix.rescale_fe_hermite_values(grad_fix, 
                                       *this, 
                                       mapping_internal_herm, 
                                       output_data.shape_gradients);
  }

  if ((flags & update_hessians) &&
      (cell_similarity != CellSimilarity::translation))
    {
      for (unsigned int k = 0; k < this->n_dofs_per_cell(); ++k)
        mapping.transform(make_array_view(fe_data.shape_hessians, k),
                          mapping_covariant_gradient,
                          mapping_internal,
                          make_array_view(output_data.shape_hessians, k));
  
      internal::Rescaler<dim, spacedim, Tensor<2,spacedim>> hessian_fix;
      hessian_fix.rescale_fe_hermite_values(hessian_fix, 
                                            *this, 
                                            mapping_internal_herm, 
                                            output_data.shape_hessians);
    }

  if ((flags & update_3rd_derivatives) &&
      (cell_similarity != CellSimilarity::translation))
    {
      for (unsigned int k = 0; k < this->n_dofs_per_cell(); ++k)
        mapping.transform(make_array_view(fe_data.shape_3rd_derivatives, k),
                          mapping_covariant_hessian,
                          mapping_internal,
                          make_array_view(output_data.shape_3rd_derivatives,
                                          k));
  
      internal::Rescaler<dim, spacedim, Tensor<3,spacedim>> third_dev_fix;
      third_dev_fix.rescale_fe_hermite_values(third_dev_fix, 
                                              *this, 
                                              mapping_internal_herm, 
                                              output_data.shape_3rd_derivatives);
    }
}
//TODO: Create a fill_fe_face_values function for Hermite elements
template <int dim, int spacedim>
void
FE_Hermite<dim, spacedim>::fill_fe_face_values(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell,
  const unsigned int                                          face_no,
  const hp::QCollection<dim - 1> &                                 quadrature,
  const Mapping<dim, spacedim> &                              mapping,
  const typename Mapping<dim, spacedim>::InternalDataBase &   mapping_internal,
  const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                     spacedim>
    &                                                            ,
  const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
  dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                     spacedim>
    &output_data) const
{
  // convert data object to internal
  // data for this class. fails with
  // an exception if that is not
  // possible
  Assert((dynamic_cast<const typename FE_Hermite<dim,spacedim>::InternalData *>(&fe_internal) != nullptr), ExcInternalError());
  const typename FE_Hermite<dim,spacedim>::InternalData &fe_data = static_cast<const typename FE_Hermite<dim,spacedim>::InternalData &>(fe_internal);
  Assert((dynamic_cast<const typename MappingHermite<dim, spacedim>::InternalData *>(&mapping_internal) != nullptr), ExcInternalError());
  const typename MappingHermite<dim, spacedim>::InternalData &mapping_internal_herm = static_cast<const typename MappingHermite<dim, spacedim>::InternalData &>(mapping_internal);
  AssertDimension(quadrature.size(), 1U);

  // offset determines which data set
  // to take (all data sets for all
  // faces are stored contiguously)

  const typename QProjector<dim>::DataSetDescriptor offset =
    QProjector<dim>::DataSetDescriptor::face(face_no,
                                             cell->face_orientation(face_no),
                                             cell->face_flip(face_no),
                                             cell->face_rotation(face_no),
                                             quadrature[0].size());

  const UpdateFlags flags(fe_data.update_each);

  // transform gradients and higher derivatives. we also have to copy
  // the values (unlike in the case of fill_fe_values()) since
  // we need to take into account the offsets
  if (flags & update_values)
  {
    for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
      for (unsigned int i = 0; i < quadrature[0].size(); ++i)
        output_data.shape_values(k, i) = fe_data.shape_values[k][i + offset];
      
    internal::Rescaler<dim, spacedim, double> shape_face_fix;
    shape_face_fix.rescale_fe_hermite_values(shape_face_fix,
                                             *this, 
                                             mapping_internal_herm, 
                                             output_data.shape_values);
  }

  if (flags & update_gradients)
  {
    for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
      mapping.transform(
        make_array_view(fe_data.shape_gradients, k, offset, quadrature[0].size()),
        mapping_covariant,
        mapping_internal,
        make_array_view(output_data.shape_gradients, k));
    
    internal::Rescaler<dim, spacedim, Tensor<1,spacedim>> grad_face_fix;
    grad_face_fix.rescale_fe_hermite_values(grad_face_fix,
                                            *this, 
                                            mapping_internal_herm, 
                                            output_data.shape_gradients);
  }

  if (flags & update_hessians)
    {
      for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
        mapping.transform(
          make_array_view(fe_data.shape_hessians, k, offset, quadrature[0].size()),
          mapping_covariant_gradient,
          mapping_internal,
          make_array_view(output_data.shape_hessians, k));

      internal::Rescaler<dim, spacedim, Tensor<2,spacedim>> hessian_face_fix;
      hessian_face_fix.rescale_fe_hermite_values(hessian_face_fix,
                                                 *this, 
                                                 mapping_internal_herm, 
                                                 output_data.shape_hessians);
    }

  if (flags & update_3rd_derivatives)
    {
      for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
        mapping.transform(make_array_view(fe_data.shape_3rd_derivatives,
                                          k,
                                          offset,
                                          quadrature[0].size()),
                          mapping_covariant_hessian,
                          mapping_internal,
                          make_array_view(output_data.shape_3rd_derivatives,
                                          k));

      internal::Rescaler<dim, spacedim, Tensor<3,spacedim>> shape_3rd_face_fix;
      shape_3rd_face_fix.rescale_fe_hermite_values(shape_3rd_face_fix,
                                                   *this, 
                                                   mapping_internal_herm, 
                                                   output_data.shape_3rd_derivatives);
    }
}


//TODO: Implement partial grid refinement (ie hanging nodes)
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

namespace VectorTools
{
    // internal namespace for implement Hermite boundary projection methods
    namespace internal
    {
        template <int dim, int spacedim = dim, typename Number = double>
        void
        get_constrained_hermite_boundary_dofs(const DoFHandler<dim, spacedim> &                                      dof_handler,
                                              const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                                              const unsigned int                                                     position,
                                              std::vector<types::global_dof_index> &                                 dof_to_boundary,
                                              types::global_dof_index &                                              no_constrained_dofs)
        {
            AssertDimension(dof_handler.n_dofs(), dof_to_boundary.size());
            std::fill_n(dof_to_boundary.begin(), dof_handler.n_dofs(), numbers::invalid_dof_index);
            
            //Create a look-up table for finding constrained dofs on all 4 or six faces of reference cell
            //TODO: Rewrite so it doesn't assume dim <= 3
            const unsigned int degree = dof_handler.get_fe().degree;
            const unsigned int regularity = ((dim == 2) ? 
                                             std::sqrt(dof_handler.get_fe().n_dofs_per_vertex()) : 
                                             std::cbrt(dof_handler.get_fe().n_dofs_per_vertex()) ) - 1;
            const unsigned int dofs_per_face = dof_handler.get_fe().n_dofs_per_face();
            const unsigned int constrained_dofs_per_face = dofs_per_face / (regularity + 1);
            
            AssertDimension(dofs_per_face, (regularity + 1) * std::pow(degree + 1, dim - 1));
            
            std::vector<types::global_dof_index> dofs_on_face(dofs_per_face);
            Table<2, double> constrained_to_local_indices(2 * dim, constrained_dofs_per_face);
            
            //Use knowledge of the local degree numbering for this version, saving expensive calls to reinit
            const std::vector<unsigned int> l2h = dealii::internal::hermite_lexicographic_to_hierarchic_numbering<dim>(regularity, degree - 2*regularity - 1);
            const auto test_cell = dof_handler.begin();

            for (unsigned int d = 0, batch_size = 1; d < dim; ++d, batch_size *= degree+1)
                for (unsigned int i = 0; i < constrained_dofs_per_face; ++i)
                {
                    const unsigned int local_index = i % batch_size;
                    const unsigned int batch_index = i / batch_size;
                    
                    unsigned int index = local_index + (batch_index * (regularity + 1) + position) * batch_size;
                    Assert(index < dofs_per_face, ExcDimensionMismatch(index, dofs_per_face));
                
                    test_cell->face(2*d)->get_dof_indices(dofs_on_face, test_cell->active_fe_index());
                    constrained_to_local_indices(2*d, i) = l2h[index];//dofs_on_face[index];
                
                    test_cell->face(2*d + 1)->get_dof_indices(dofs_on_face, test_cell->active_fe_index());
                    constrained_to_local_indices(2*d + 1,i) = l2h[index];//dofs_on_face[index];
                }

            std::set<types::boundary_id> selected_boundary_components;
            for (auto i = boundary_functions.cbegin(); i != boundary_functions.cend(); ++i)
                selected_boundary_components.insert(i->first);
            
            Assert(selected_boundary_components.find(numbers::internal_face_boundary_id) ==
                    selected_boundary_components.end(), DoFTools::ExcInvalidBoundaryIndicator());
            
            no_constrained_dofs = 0;
            
            for (const auto &cell : dof_handler.active_cell_iterators())
                for (const unsigned int f : cell->face_indices())
                {
                    AssertDimension(cell->get_fe().n_dofs_per_face(f), dofs_on_face.size());
                    
                    //Check if face is on selected boundary section
                    if (selected_boundary_components.find(cell->face(f)->boundary_id()) !=
                        selected_boundary_components.end())
                    {
                        cell->face(f)->get_dof_indices(dofs_on_face, cell->active_fe_index());
                        
                        for (unsigned int i = 0; i < constrained_dofs_per_face; ++i)
                        {
                            const types::global_dof_index index = dofs_on_face[constrained_to_local_indices(f, i)];
                            Assert(index < dof_to_boundary.size(), ExcDimensionMismatch(index, dof_to_boundary.size()));
                            
                            if (dof_to_boundary[index] == numbers::invalid_dof_index)
                                dof_to_boundary[index] = no_constrained_dofs++;
                        }
                    }
                }
            
            Assert((no_constrained_dofs != dof_handler.n_boundary_dofs(boundary_functions)) || (regularity == 0),
                   ExcInternalError());
        }
                                
        
        template <int dim, int spacedim = dim, typename Number = double>
        void
        do_hermite_direct_projection(const MappingHermite<dim, spacedim> &                             mapping_h,
                                const DoFHandler<dim, spacedim> &                                      dof_handler,
                                const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                                const Quadrature<dim - 1> &                                            quadrature,
                                const unsigned int                                                     position,
                                std::map<types::global_dof_index, Number> &                            boundary_values,
                                std::vector<unsigned int>                                              component_mapping = {})
        {
            //Return immediately if no constraint functions are provided
            if (boundary_functions.size() == 0) return;
            
            //For dim=1, the problem simplifies to interpolation at the boundaries
            if (dim == 1)
            {
                Assert(component_mapping.size() == 0, ExcNotImplemented());
                for (const auto &cell : dof_handler.active_cell_iterators())
                {
                    for (const unsigned int direction : GeometryInfo<dim>::face_indices())
                    {
                        if (cell->at_boundary(direction) &&
                            boundary_functions.find(cell->face(direction)->boundary_id()) != boundary_functions.end())
                        {
                            const Function<spacedim, Number> & current_function =
                                *boundary_functions.find(cell->face(direction)->boundary_id())->second;
                                
                            const FiniteElement<dim, spacedim> &fe_herm = dof_handler.get_fe();
                            AssertDimension(fe_herm.n_components(), current_function.n_components);
                            
                            Vector<Number> boundary_value(fe_herm.n_components());
                            if (boundary_value.size() == 1)
                                boundary_value(0) = current_function.value(cell->vertex(direction));
                            else
                                current_function.vector_value(cell->vertex(direction), boundary_value);
                            
                            boundary_values[cell->vertex_dof_index(direction, position, cell->active_fe_index())] =
                                boundary_value(fe_herm.face_system_to_component_index(position).first);
                        }
                    }
                }
                return;
            }
            
            //dim=2 or higher, needs actual projection
            if (component_mapping.size() == 0)
            {
                AssertDimension(dof_handler.get_fe().n_components(), boundary_functions.begin()->second->n_components);
                component_mapping.resize(dof_handler.get_fe().n_components());
                for (unsigned int i = 0; i < component_mapping.size(); ++i)
                    component_mapping[i] = i;
            }
            else AssertDimension(component_mapping.size(), dof_handler.get_fe().n_components());
            
            std::vector<types::global_dof_index> dof_to_boundary_mapping(dof_handler.n_dofs(), numbers::invalid_dof_index);
            types::global_dof_index next_boundary_index;
            
            get_constrained_hermite_boundary_dofs(dof_handler, boundary_functions, position, dof_to_boundary_mapping, next_boundary_index);
            
            SparsityPattern sparsity;
            {
                DynamicSparsityPattern dsp(next_boundary_index, next_boundary_index);
                DoFTools::make_boundary_sparsity_pattern(dof_handler,
                                                         boundary_functions,
                                                         dof_to_boundary_mapping,
                                                         dsp);
            
                sparsity.copy_from(dsp);
            }
            
            //Assert mesh is not partially refined
            //TODO: Add functionality on partially refined meshes
            int level = -1;
            for (const auto &cell : dof_handler.active_cell_iterators())
                for (auto f : cell->face_indices())
                {
                    if (cell->at_boundary(f))
                    {
                        if (level == -1) level = cell->level();
                        else Assert(level == cell->level(),
                                    ExcMessage("The mesh you use in projecting boundary values "
                                               "has hanging nodes at the boundary. This would require "
                                               "dealing with hanging node constraints when solving "
                                               "the linear system on the boundary, but this is not "
                                               "currently implemented."));
                    }
                }
            
            // make mass matrix and right hand side
            SparseMatrix<Number> mass_matrix(sparsity);
            Vector<Number>       rhs(sparsity.n_rows());
            Vector<Number>       boundary_projection(rhs.size());
            
            MatrixCreator::create_boundary_mass_matrix(mapping_h,
                                                       dof_handler,
                                                       quadrature,
                                                       mass_matrix,
                                                       boundary_functions,
                                                       rhs,
                                                       dof_to_boundary_mapping,
                                                       static_cast<const Function<spacedim, Number> *>(nullptr),
                                                       component_mapping);

      if (rhs.norm_sqr() < 1e-8)
        boundary_projection = 0;
      else
        {
          // Allow for a maximum of 5*n steps to reduce the residual by 10^-12.
          // n steps may not be sufficient, since roundoff errors may accumulate
          // for badly conditioned matrices
          ReductionControl control(5 * rhs.size(), 0., 1e-12, false, false);
          GrowingVectorMemory<Vector<Number>> memory;
          SolverCG<Vector<Number>>            cg(control, memory);

          PreconditionSSOR<SparseMatrix<Number>> prec;
          prec.initialize(mass_matrix, 1.2);

          cg.solve(mass_matrix, boundary_projection, rhs, prec);
          
          deallog << "Number of CG iterations in project boundary values: " << control.last_step() << std::endl;
        }
      // fill in boundary values
      for (unsigned int i = 0; i < dof_to_boundary_mapping.size(); ++i)
        if (dof_to_boundary_mapping[i] != numbers::invalid_dof_index)
          {
            AssertIsFinite(boundary_projection(dof_to_boundary_mapping[i]));

            // this dof is on one of the
            // interesting boundary parts
            //
            // remember: i is the global dof
            // number, dof_to_boundary_mapping[i]
            // is the number on the boundary and
            // thus in the solution vector
            boundary_values[i] =
              boundary_projection(dof_to_boundary_mapping[i]);
          }
        }
    }   //namespace internal
    
    template <int dim, int spacedim, typename Number>
    void
    project_boundary_values(const MappingHermite<dim, spacedim> &                                  mapping_h,
                            const DoFHandler<dim, spacedim> &                                      dof_handler,
                            const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                            const Quadrature<dim - 1> &                                            quadrature,
                            const HermiteBoundaryType                                              projection_mode,
                            std::map<types::global_dof_index, Number> &                            boundary_values,
                            std::vector<unsigned int>                                              component_mapping)
    {
        //This version implements projected values directly, so it's necessary to check that this is possible
        const bool check_mode = (projection_mode == HermiteBoundaryType::hermite_dirichlet) ||
                                (projection_mode == HermiteBoundaryType::hermite_neumann) ||
                                (projection_mode == HermiteBoundaryType::hermite_2nd_derivative);
        Assert(check_mode, ExcNotImplemented());
        (void)check_mode;
        
        unsigned int position = 0;
        switch(projection_mode)
        {
            case HermiteBoundaryType::hermite_dirichlet:
                position = 0;
                break;
            case HermiteBoundaryType::hermite_neumann:
                Assert(dof_handler.get_fe().degree > 2, ExcDimensionMismatch(dof_handler.get_fe().degree, 3));
                position = 1;
                break;
            case HermiteBoundaryType::hermite_2nd_derivative:
                Assert(dof_handler.get_fe().degree > 4, ExcDimensionMismatch(dof_handler.get_fe().degree, 5));
                position = 2;
                break;
            default:
                Assert(false, ExcInternalError());
        }
        
        internal::do_hermite_direct_projection(mapping_h, dof_handler, boundary_functions, quadrature, position, boundary_values, component_mapping);
    }
    
    template <int dim, int spacedim, typename Number>
    void
    project_boundary_values(const MappingHermite<dim, spacedim> &                                  mapping_h,
                            const DoFHandler<dim, spacedim> &                                      dof_handler,
                            const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                            const Quadrature<dim - 1> &                                            quadrature,
                            std::map<types::global_dof_index, Number> &                            boundary_values,
                            std::vector<unsigned int>                                              component_mapping)
    {
        internal::do_hermite_direct_projection(mapping_h, dof_handler, boundary_functions, quadrature, 0, boundary_values, component_mapping);
    }
    
    template <int dim, int spacedim, typename Number>
    void
    project_boundary_values(const MappingHermite<dim, spacedim> &                                  mapping_h,
                            const DoFHandler<dim, spacedim> &                                      dof_handler,
                            const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                            const Quadrature<dim - 1> &                                            quadrature,
                            const HermiteBoundaryType                                              projection_mode,
                            AffineConstraints<Number>                                              constraints,
                            std::vector<unsigned int>                                              component_mapping)
    {
        Assert(false, ExcNotImplemented());
    }
    
    template <int dim, int spacedim, typename Number>
    void
    project_boundary_values(const MappingHermite<dim, spacedim> &                                  mapping_h,
                            const DoFHandler<dim, spacedim> &                                      dof_handler,
                            const std::map<types::boundary_id, const Function<spacedim, Number>*> &boundary_functions,
                            const Quadrature<dim - 1> &                                            quadrature,
                            AffineConstraints<Number>                                              constraints,
                            std::vector<unsigned int>                                              component_mapping)
    {
        Assert(false, ExcNotImplemented());
    }
    
    template <int dim, typename VectorType, int spacedim> void
    project(const MappingHermite<dim, spacedim> &                     mapping,
            const DoFHandler<dim, spacedim> &                         dof,
            const AffineConstraints<typename VectorType::value_type> &constraints,
            const Quadrature<dim> &                                   quadrature,
            const Function<spacedim, typename VectorType::value_type> &function,
            VectorType &                                               vec,
            const bool                 enforce_zero_boundary,
            const Quadrature<dim - 1> &q_boundary,
            const bool                 project_to_boundary_first = false)
    {
        using number = typename VectorType::value_type;
        Assert((dynamic_cast<const FE_Hermite<dim, spacedim>*>( &dof.get_fe() ) != nullptr),
               ExcMessage("The project function with the MappingHermite mapping class "
                          "should only be used with a DoF handler associated with an "
                          "FE_Hermite object with the same dim and spacedim as the mapping."));
        Assert(dof.get_fe(0).n_components() == function.n_components,
               ExcDimensionMismatch(dof.get_fe(0).n_components(),
                                    function.n_components));
        Assert(vec.size() == dof.n_dofs(),
               ExcDimensionMismatch(vec.size(), dof.n_dofs()));
        
        // make up boundary values
        std::map<types::global_dof_index, number> boundary_values;
        const std::vector<types::boundary_id> active_boundary_ids = dof.get_triangulation().get_boundary_ids();
        
        if (enforce_zero_boundary)
        {
            std::map<types::boundary_id, Function<spacedim, number>*> boundary;
            for (const auto it : active_boundary_ids)
                boundary.emplace(std::make_pair(it, nullptr));
            
            std::vector<types::global_dof_index> dof_to_boundary(dof_handler.n_dofs(), numbers::invalid_dof_index);
            types::global_dof_index end_boundary_dof = 0;
            
            internal::get_constrained_hermite_boundary_dofs(dof_handler, boundary, 0, dof_to_boundary, end_boundary_dof);
            
            if (end_boundary_dof == 0) break;
            
            for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
                if (dof_to_boundary[i] != numbers::invalid_dof_index)
                    boundary_values.emplace(std::make_pair(i, 0));
        }
        else if (project_to_boundary_first)
        {
            std::map<types::boundary_id, Function<spacedim, number>*> boundary_function;
            for (const auto it : active_boundary_ids)
                boundary_function.emplace(std::make_pair(it, &function));
            
            project_boundary_values<dim, spacedim, number>(mapping,
                                                           dof,
                                                           boundary_function,
                                                           q_boundary,
                                                           HermiteBoundaryType::hermite_dirichlet,
                                                           boundary_values);
        }
        
        // check if constraints are compatible (see below)
        bool constraints_are_compatible = true;
        for (const auto &value : boundary_values)
            if (constraints.is_constrained(value.first())
                if ((constraints.get_constraint_entries(value.first())->size > 0) &&
                    (constraints.get_inhomogeneity(value.first()) != value.second))
                    constraints_are_compatible = false;
        
        //TODO: Continue re-writing function from here!
        //TODO: resolve confusion about vec and vec_result
                
        // set up mass matrix and right hand side
        Vector<number>  vec(dof.n_dofs());
        SparsityPattern sparsity;
        
        {
            DynamicSparsityPattern dsp(dof.n_dofs(), dof.n_dofs());
            DoFTools::make_sparsity_pattern(dof,
                                            dsp,
                                            constraints,
                                            !constraints_are_compatible);

            sparsity.copy_from(dsp);
        }
        
        SparseMatrix<number> mass_matrix(sparsity);
        Vector<number>       tmp(mass_matrix.n());

        // If the constraints object does not conflict with the given boundary
        // values (i.e., it either does not contain boundary values or it contains
        // the same as boundary_values), we can let it call
        // distribute_local_to_global straight away, otherwise we need to first
        // interpolate the boundary values and then condense the matrix and vector
        if (constraints_are_compatible)
        {
            const Function<spacedim, number> *dummy = nullptr;
            MatrixCreator::create_mass_matrix(mapping,
                                              dof,
                                              quadrature,
                                              mass_matrix,
                                              function,
                                              tmp,
                                              dummy,
                                              constraints);
            
            if (boundary_values.size() > 0)
                MatrixTools::apply_boundary_values(boundary_values, mass_matrix, vec, tmp, true);
        }
        else
        {
            // create mass matrix and rhs at once, which is faster.
            MatrixCreator::create_mass_matrix(mapping, dof, quadrature, mass_matrix, function, tmp);
            MatrixTools::apply_boundary_values(boundary_values, mass_matrix, vec, tmp, true);
            constraints.condense(mass_matrix, tmp);
        }

        // Allow for a maximum of 5*n steps to reduce the residual by 10^-12. n
        // steps may not be sufficient, since roundoff errors may accumulate for
        // badly conditioned matrices
        ReductionControl control(5 * tmp.size(), 0., 1e-12, false, false);
        GrowingVectorMemory<Vector<number>> memory;
        SolverCG<Vector<number>>            cg(control, memory);

        PreconditionSSOR<SparseMatrix<number>> prec;
        prec.initialize(mass_matrix, 1.2);

        cg.solve(mass_matrix, vec, tmp, prec);
        constraints.distribute(vec);

        // copy vec into vec_result. we can't use vec_result itself above, since
        // it may be of another type than Vector<double> and that wouldn't
        // necessarily go together with the matrix and other functions
        for (unsigned int i = 0; i < vec.size(); ++i)
            ::dealii::internal::ElementAccess<VectorType>::set(vec(i),
                                                               i,
                                                               vec_result);
    }
} //namespace VectorTools




//Explicit instantiations
#include "fe_hermite.inst.in"

DEAL_II_NAMESPACE_CLOSE
