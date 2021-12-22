// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/base/template_constraints.h>
#include <deal.II/base/table.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_hermite.h>
#include <deal.II/fe/fe_hermite_customregularity.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/matrix_tools.h>

#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <iostream>
#include <algorithm>

DEAL_II_NAMESPACE_OPEN



/**
 * Source code for Hermite basis functions.
 */

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
        
        AssertDimension(count, dim_dofs_1d * dim_dofs_2d);
  }



  template <>
  void
  hermite_hierarchic_to_lexicographic_numbering<4>(const unsigned int regularity,
                                                   const unsigned int nodes,
                                                   std::vector<unsigned int> &h2l)
  {
        Assert(false, ExcNotImplemented());
        
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



  static inline unsigned int
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
  class RescalerCustomreg
  {
  public:
      void
      rescale_fe_hermite_values(const FE_CustomHermite<xdim, xspacedim> &                     fe_herm,
                                const typename MappingHermite<xdim, xspacedim>::InternalData &mapping_data,
                                Table<2, xNumber> &                                           value_list);
  
      
      
      //TODO: Check these functions work with DOFs assigned on edges, faces etc
      template <int spacedim, typename Number>
      void 
      rescale_fe_hermite_values(RescalerCustomreg<1, spacedim, Number> &                  /*rescaler*/,
                                const FE_CustomHermite<1, spacedim> &                     fe_herm,
                                const typename MappingHermite<1, spacedim>::InternalData &mapping_data,
                                Table<2, Number> &                                        value_list)
      {
          const unsigned int dofs = fe_herm.n_dofs_per_cell();
          const unsigned int regularity = fe_herm.get_regularity();
          const unsigned int nodes = fe_herm.n_nodes();
          const unsigned int q_points = value_list.size(1);
          (void)dofs;
          
          AssertDimension(value_list.size(0), dofs);
          AssertDimension(dofs, 2 * regularity + nodes + 2);
          AssertIndexRange(q_points, mapping_data.quadrature_points.size() + 1);
        
          std::vector<unsigned int> l2h = hermite_lexicographic_to_hierarchic_numbering<1>(regularity, nodes);
          
          for (unsigned int q = 0; q < q_points; ++q)
          {
              double factor_1 = 1.0;
            
              for (unsigned int d1 = 0, d2 = regularity + nodes + 1; d2 < dofs; ++d1, ++d2)
              {
                  value_list(l2h[d1], q) *= factor_1;
                  value_list(l2h[d2], q) *= factor_1;
                  
                  factor_1 *= mapping_data.cell_extents[0]; 
              }
          }
      }
      
      
      
      template <int spacedim, typename Number>
      void 
      rescale_fe_hermite_values(RescalerCustomreg<2, spacedim, Number> &                  /*rescaler*/,
                                const FE_CustomHermite<2, spacedim> &                     fe_herm,
                                const typename MappingHermite<2, spacedim>::InternalData &mapping_data,
                                Table<2, Number> &                                        value_list)
      {
          const unsigned int dofs = fe_herm.n_dofs_per_cell();
          const unsigned int regularity = fe_herm.get_regularity();
          const unsigned int nodes = fe_herm.n_nodes();
          const unsigned int dofs_per_dim = 2 * regularity + nodes + 2;
          const unsigned int q_points = value_list.size(1);
          (void)dofs;
          
          AssertDimension(value_list.size(0), dofs);
          AssertDimension(dofs_per_dim * dofs_per_dim, dofs);
          AssertIndexRange(q_points, mapping_data.quadrature_points.size() + 1);
          
          std::vector<unsigned int> l2h = hermite_lexicographic_to_hierarchic_numbering<2>(regularity, nodes);
      
          AssertDimension(l2h.size(), dofs);
          
          FullMatrix<double> factors(dofs_per_dim, dofs_per_dim);
          for (unsigned int q = 0; q < q_points; ++q)
          {
              double factor_2 = 1.0;
                    
              for (unsigned int d3 = 0, d4 = regularity + nodes + 1; d4 < dofs_per_dim; ++d3, ++d4)
              {
                  double factor_1 = factor_2;
                
                  for (unsigned int d1 = 0, d2 = regularity + nodes + 1; d2 < dofs_per_dim; ++d1, ++d2)
                  {
                      factors(d1,d3) = factor_1;
                      factors(d2,d3) = factor_1;
                      factors(d1,d4) = factor_1;
                      factors(d2,d4) = factor_1;
                      
                      value_list(l2h[d1 + d3 * dofs_per_dim], q) *= factor_1;
                      value_list(l2h[d2 + d3 * dofs_per_dim], q) *= factor_1;
                      value_list(l2h[d1 + d4 * dofs_per_dim], q) *= factor_1;
                      value_list(l2h[d2 + d4 * dofs_per_dim], q) *= factor_1;
                            
                      factor_1 *= mapping_data.cell_extents[0];
                  }
                
                  factor_2 *= mapping_data.cell_extents[1];
              }
          }
      }
      
      
  
      template <int spacedim, typename Number>
      void
      rescale_fe_hermite_values(RescalerCustomreg<3, spacedim, Number> &                  /*rescaler*/,
                                const FE_CustomHermite<3, spacedim> &                     fe_herm,
                                const typename MappingHermite<3, spacedim>::InternalData &mapping_data,
                                Table<2, Number> &                                        value_list)
      {
          const unsigned int dofs = fe_herm.n_dofs_per_cell();
          const unsigned int regularity = fe_herm.get_regularity();
          const unsigned int nodes = fe_herm.n_nodes();
          const unsigned int dofs_per_dim = 2 * regularity + nodes + 2;
          const unsigned int q_points = value_list.size(1);
          (void)dofs;
          
          AssertDimension(value_list.size(0), dofs);
          AssertDimension(dofs_per_dim * dofs_per_dim * dofs_per_dim, dofs);
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
  };    //class RescalerCustomreg
}   //namespace internal
  
  

//TODO: re-write the following functions for adaptive grid refinement with FE_CustomHermite
/*
 * Implementation structs for providing constraint matrices for hanging nodes
 */
  static void
  create_F_matrix(std::vector<double> &      F_matrix,
                  const unsigned int         regularity,
                  const unsigned int         nodes)
  {
      Assert(false, ExcNotImplemented());
  }



  static inline void
  create_G_matrix(std::vector<double> &      G_matrix,
                  const std::vector<double> &F_matrix,
                  const unsigned int         regularity,
                  const unsigned int         nodes)
  {
      Assert(false, ExcNotImplemented());
  }



  template <int spacedim>
  static void
  initialise_constraints(FE_CustomHermite<1, spacedim> &fe_hermite)
  {
    Assert(false, ExcNotImplemented());
  }
  
  

  template <int spacedim>
  static void
  initialise_constraints(FE_CustomHermite<2, spacedim> &fe_hermite)
  {
    Assert(false, ExcNotImplemented());
  }
  
  

  template <int spacedim>
  static void
  initialise_constraints(FE_CustomHermite<3, spacedim> &fe_hermite)
  {
      Assert(false, ExcNotImplemented());
  }
};  //struct FE_CustomHermite::Implementation



/*
 * Member functions for the Hermite class
 */
//Constructors
template <int dim, int spacedim>
FE_CustomHermite<dim, spacedim>::FE_CustomHermite(const unsigned int reg, const unsigned int nodes)
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



template <int dim, int spacedim>
void
FE_CustomHermite<dim, spacedim>::initialize_constraints()
{
  Implementation::initialise_constraints(*this);
  return;
}



template <int dim, int spacedim>
std::string
FE_CustomHermite<dim, spacedim>::get_name() const
{
  std::ostringstream name_buffer;
  name_buffer << "FE_CustomHermite<" << dim << "," << spacedim << ">("
              << this->regularity << "," << this->nodes << ")";
  return name_buffer.str();
}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_CustomHermite<dim, spacedim>::clone() const
{
  return std::make_unique<FE_CustomHermite<dim, spacedim>>(*this);
}



template <int dim, int spacedim>
void
FE_CustomHermite<dim, spacedim>::fill_fe_values(
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
  /*
   * convert data object to internal data for this class. 
   * Fails with an exception if that is not possible.
   */
  Assert((dynamic_cast<const typename FE_CustomHermite<dim,spacedim>::InternalData *>(&fe_internal) != nullptr),
         ExcInternalError());
  const typename FE_CustomHermite<dim,spacedim>::InternalData &fe_data 
        = static_cast<const typename FE_CustomHermite<dim,spacedim>::InternalData &>(fe_internal);
  
  Assert((dynamic_cast<const typename MappingHermite<dim, spacedim>::InternalData *>(&mapping_internal) != nullptr),
         ExcInternalError());
  const typename MappingHermite<dim, spacedim>::InternalData &mapping_internal_herm 
        = static_cast<const typename MappingHermite<dim, spacedim>::InternalData &>(mapping_internal);

  const UpdateFlags flags(fe_data.update_each);

  /*
   * Transform values gradients and higher derivatives. Values need to
   * be rescaled according the the nodal derivative they correspond to.
   */
  if ((flags & update_values) &&
      (cell_similarity != CellSimilarity::translation))
  {
      internal::RescalerCustomreg<dim, spacedim, double> shape_fix;
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
    
    internal::RescalerCustomreg<dim, spacedim, Tensor<1,spacedim>> grad_fix;
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
  
      internal::RescalerCustomreg<dim, spacedim, Tensor<2,spacedim>> hessian_fix;
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
  
      internal::RescalerCustomreg<dim, spacedim, Tensor<3,spacedim>> third_dev_fix;
      third_dev_fix.rescale_fe_hermite_values(third_dev_fix, 
                                              *this, 
                                              mapping_internal_herm, 
                                              output_data.shape_3rd_derivatives);
    }
}



template <int dim, int spacedim>
void
FE_CustomHermite<dim, spacedim>::fill_fe_face_values(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell,
  const unsigned int                                          face_no,
  const hp::QCollection<dim - 1> &                            quadrature,
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
  /*
   * Convert data object to internal data for this class. Fails with
   * an exception if that is not possible.
   */
  Assert((dynamic_cast<const typename FE_CustomHermite<dim,spacedim>::InternalData *>(&fe_internal) != nullptr), ExcInternalError());
  const typename FE_CustomHermite<dim,spacedim>::InternalData &fe_data 
        = static_cast<const typename FE_CustomHermite<dim,spacedim>::InternalData &>(fe_internal);
  
  Assert((dynamic_cast<const typename MappingHermite<dim, spacedim>::InternalData *>(&mapping_internal) != nullptr), ExcInternalError());
  const typename MappingHermite<dim, spacedim>::InternalData &mapping_internal_herm 
        = static_cast<const typename MappingHermite<dim, spacedim>::InternalData &>(mapping_internal);
  
  AssertDimension(quadrature.size(), 1U);

  /*
   * offset determines which data set to take (all data sets for all
   * faces are stored contiguously)
   */
  const typename QProjector<dim>::DataSetDescriptor offset =
    QProjector<dim>::DataSetDescriptor::face(face_no,
                                             cell->face_orientation(face_no),
                                             cell->face_flip(face_no),
                                             cell->face_rotation(face_no),
                                             quadrature[0].size());

  const UpdateFlags flags(fe_data.update_each);

  /*
   * Transform gradients and higher derivatives. we also have to copy
   * the values (unlike in the case of fill_fe_values()) since
   * we need to take into account the offsets
   */
  if (flags & update_values)
  {
    for (unsigned int k = 0; k < this->dofs_per_cell; ++k)
      for (unsigned int i = 0; i < quadrature[0].size(); ++i)
        output_data.shape_values(k, i) = fe_data.shape_values[k][i + offset];
      
    internal::RescalerCustomreg<dim, spacedim, double> shape_face_fix;
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
    
    internal::RescalerCustomreg<dim, spacedim, Tensor<1,spacedim>> grad_face_fix;
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

      internal::RescalerCustomreg<dim, spacedim, Tensor<2,spacedim>> hessian_face_fix;
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

      internal::RescalerCustomreg<dim, spacedim, Tensor<3,spacedim>> shape_3rd_face_fix;
      shape_3rd_face_fix.rescale_fe_hermite_values(shape_3rd_face_fix,
                                                   *this, 
                                                   mapping_internal_herm, 
                                                   output_data.shape_3rd_derivatives);
    }
}



//TODO: Implement partial grid refinement (ie hanging nodes)
/*
template <int dim, int spacedim>
void FE_CustomHermite<dim, spacedim>::get_interpolation_matrix(const
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
    /*
     * Internal namespace for implement Hermite boundary projection methods
     */
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
            const unsigned int degree = dof_handler.get_fe().degree;
            const unsigned int regularity = dynamic_cast<const FE_CustomHermite<dim>&>(dof_handler.get_fe()).get_regularity();
            const unsigned int dofs_per_face = dof_handler.get_fe().n_dofs_per_face();
            const unsigned int constrained_dofs_per_face = dofs_per_face / (regularity + 1);
            
            AssertDimension(dofs_per_face, (regularity + 1) * std::pow(degree + 1, dim - 1));
            
            std::vector<types::global_dof_index> dofs_on_face(dofs_per_face);
            Table<2, double> constrained_to_local_indices(2 * dim, constrained_dofs_per_face);
            
            //Use knowledge of the local degree numbering for this version, saving expensive calls to reinit
            const std::vector<unsigned int> l2h 
                = dealii::internal::hermite_lexicographic_to_hierarchic_numbering<dim>(regularity, degree - 2*regularity - 1);
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
                    for (const unsigned int direction : GeometryInfo<dim>::face_indices())
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
          /*
           * Allow for a maximum of 5*n steps to reduce the residual by 10^-12.
           * n steps may not be sufficient, since roundoff errors may accumulate
           * for badly conditioned matrices
           */
          /*
            ReductionControl control(5 * rhs.size(), 0., 1e-12, false, false);
          GrowingVectorMemory<Vector<Number>> memory;
          SolverCG<Vector<Number>>            cg(control, memory);

          PreconditionSSOR<SparseMatrix<Number>> prec;
          prec.initialize(mass_matrix, 1.2);

          cg.solve(mass_matrix, boundary_projection, rhs, prec);
          deallog << "Number of CG iterations in project boundary values: " << control.last_step() << std::endl;
          */
          SparseDirectUMFPACK mass_inv;
          mass_inv.initialize(mass_matrix);
          mass_inv.vmult(boundary_projection, rhs);
        }
        
      // fill in boundary values
      for (unsigned int i = 0; i < dof_to_boundary_mapping.size(); ++i)
        if (dof_to_boundary_mapping[i] != numbers::invalid_dof_index)
          {
            AssertIsFinite(boundary_projection(dof_to_boundary_mapping[i]));

            /*
             * This dof is on one of the interesting boundary parts
             * 
             * Remember: i is the global dof number, dof_to_boundary_mapping[i]
             * is the number on the boundary and thus in the solution vector
             */
            boundary_values[i] = boundary_projection(dof_to_boundary_mapping[i]);
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
        Assert((dynamic_cast<const FE_CustomHermite<dim>*>(&dof_handler.get_fe()) != nullptr),
                ExcMessage("The version of project boundary functions with a MappingHermite object as the mapping argument"
                           " should only be used with a DOF handler linked to an FE_CustomHermite object."));
        
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
    project(const MappingHermite<dim, spacedim> &                      mapping,
            const DoFHandler<dim, spacedim> &                          dof,
            const AffineConstraints<typename VectorType::value_type> & constraints,
            const Quadrature<dim> &                                    quadrature,
            const Function<spacedim, typename VectorType::value_type> &function,
            VectorType &                                               vec,
            const bool                                                 enforce_zero_boundary,
            const Quadrature<dim - 1> &                                q_boundary,
            const bool                                                 project_to_boundary_first)
    {
        using number = typename VectorType::value_type;
        
        Assert((dynamic_cast<const FE_CustomHermite<dim, spacedim>*>( &dof.get_fe() ) != nullptr),
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
            std::map<types::boundary_id, const Function<spacedim, number>*> boundary;
            
            for (const auto it : active_boundary_ids)
                boundary.emplace(std::make_pair(it, nullptr));
            
            const std::map<types::boundary_id, const Function<spacedim, number>* > new_boundary = 
                static_cast<const std::map<types::boundary_id, const Function<spacedim, number>*>>(boundary);
            
            std::vector<types::global_dof_index> dof_to_boundary(dof.n_dofs(), numbers::invalid_dof_index);
            types::global_dof_index end_boundary_dof = 0;
            
            internal::get_constrained_hermite_boundary_dofs(dof, new_boundary, 0, dof_to_boundary, end_boundary_dof);
            
            if (end_boundary_dof != 0)            
                for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
                    if (dof_to_boundary[i] != numbers::invalid_dof_index)
                        boundary_values.emplace(std::make_pair(i, 0));
        }
        else if (project_to_boundary_first)
        {
            std::map<types::boundary_id, const Function<spacedim, number>*> boundary_function;
            
            for (const auto it : active_boundary_ids)
                boundary_function.emplace(std::make_pair(it, &function));
            
            const std::map<types::boundary_id, const Function<spacedim, number>* > new_boundary_2 =
                static_cast<const std::map< types::boundary_id, const Function<spacedim, number>* >>(boundary_function);
            
            project_boundary_values<dim, spacedim, number>(mapping,
                                                           dof,
                                                           new_boundary_2,
                                                           q_boundary,
                                                           HermiteBoundaryType::hermite_dirichlet,
                                                           boundary_values);
        }
        
        // check if constraints are compatible (see below)
        bool constraints_are_compatible = true;
        
        for (const auto &value : boundary_values)
            if (constraints.is_constrained(value.first))
                if ((constraints.get_constraint_entries(value.first)->size() > 0) &&
                    (constraints.get_inhomogeneity(value.first) != value.second))
                    constraints_are_compatible = false;
                
        // set up mass matrix and right hand side
        Vector<number>  vec_result(dof.n_dofs());
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

        /*
         * If the constraints object does not conflict with the given boundary
         * values (i.e., it either does not contain boundary values or it contains
         * the same as boundary_values), we can let it call
         * distribute_local_to_global straight away, otherwise we need to first
         * interpolate the boundary values and then condense the matrix and vector
         */
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
                MatrixTools::apply_boundary_values(boundary_values, mass_matrix, vec_result, tmp, true);
        }
        else
        {
            // create mass matrix and rhs at once, which is faster.
            MatrixCreator::create_mass_matrix(mapping, dof, quadrature, mass_matrix, function, tmp);
            MatrixTools::apply_boundary_values(boundary_values, mass_matrix, vec_result, tmp, true);
            constraints.condense(mass_matrix, tmp);
        }

        /*
         * Allow for a maximum of 5*n steps to reduce the residual by 10^-12. n
         * steps may not be sufficient, since roundoff errors may accumulate for
         * badly conditioned matrices
         */
        /*
        ReductionControl control(5 * tmp.size(), 1e-8, 1e-8, false, false);
        GrowingVectorMemory<Vector<number>> memory;
        SolverCG<Vector<number>>            cg(control, memory);

        PreconditionSSOR<SparseMatrix<number>> prec;
        prec.initialize(mass_matrix, 1.2);

        cg.solve(mass_matrix, vec_result, tmp, prec);
        */
        
        SparseDirectUMFPACK mass_inv;
        mass_inv.initialize(mass_matrix);
        mass_inv.vmult(vec_result, tmp);
        
        constraints.distribute(vec_result);

        /*
         * copy vec_result into vec. we can't use vec itself above, since
         * it may be of another type than Vector<double> and that wouldn't
         * necessarily go together with the matrix and other functions
         */
        for (unsigned int i = 0; i < vec.size(); ++i)
            ::dealii::internal::ElementAccess<VectorType>::set(vec_result(i),
                                                               i,
                                                               vec);
    }
} //namespace VectorTools



//Explicit instantiations
#include "fe_hermite_customregularity.inst"

DEAL_II_NAMESPACE_CLOSE
