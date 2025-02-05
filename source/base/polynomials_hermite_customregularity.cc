// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2021 by the deal.II authors
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

#include <deal.II/base/polynomials_hermite.h>
#include <deal.II/base/polynomials_hermite_customregularity.h>

/* Note:
 * Several functions in this code file are sensitive to underflow error
 * when using unsigned ints for arithmetic. At present static casts to
 * int are used to fix the issue. If this bug occurs it usually affects
 * FE_Hermite(2) and higher, with shape values taking large magnitude
 * values in the interval (0,1).
 *  - Ivy Weber
 */

DEAL_II_NAMESPACE_OPEN

namespace Polynomials
{
    namespace internal 
    {
        static inline unsigned int
        factorial(const unsigned int n)
        {
        unsigned int result = 1;
        for (unsigned int i = n; i > 0; i--)
        result *= i;
        return result;
        }
    }   //internal
    
    

  Polynomial<double>
  interpolant_poly(const std::vector<Point<1>> &nodes)
  {
    Polynomial<double>  out_poly(std::vector<double>({1.0}));
    std::vector<double> temp_coeffs({0.0, 1.0});
    
    for (auto &nd : nodes)
      {
        temp_coeffs[0] = -nd(0);
        out_poly *= Polynomial<double>(temp_coeffs);
      }
      
    return out_poly;
  }
  
  

  std::vector<Point<1>>
  HermiteCustomreg::create_chebyshevgausslobatto_nodes(
    const unsigned int no_nodes)
  {
    Assert(
      no_nodes != 0,
      ExcMessage(
        "The custom-regularity Hermite class should not be used without"
        " interpolation nodes. Use the maximum regularity version instead."));
    
    std::vector<Point<1>> pts;
    double                step = M_PI / (no_nodes + 1);
    
    for (unsigned int i = no_nodes; i > 0; --i)
      pts.push_back(Point<1>(0.5 * (1.0 + std::cos(step * i))));
    
    return pts;
  }

  
  
  std::vector<double>
  HermiteCustomreg::interpolant_derivatives(const std::vector<Point<1>> &nodes)
  {
    const unsigned int n = nodes.size();
    
    Assert(
      n != 0,
      ExcMessage(
        "The custom-regularity Hermite class should not be used without"
        " interpolation nodes. Use the maximum regularity version instead."));
    
    Polynomial<double> p      = interpolant_poly(nodes);
    std::vector<double> values(n + 1);
    
    p.value(0., n, values.data());
    
    return values;
  }

  
  
  FullMatrix<double>
  HermiteCustomreg::modified_hermite_to_bernstein_matrix(
    const unsigned int           regularity,
    const std::vector<Point<1>> &nodes)
  {
    Assert(
      nodes.size() != 0,
      ExcMessage(
        "The custom-regularity Hermite class should not be used without"
        " interpolation nodes. Use the maximum regularity version instead."));
    
    std::vector<double> deriv_vals = interpolant_derivatives(nodes);
    deriv_vals.resize(regularity + 1);
    
    FullMatrix<double> B_std =
      HermitePoly::hermite_to_bernstein_matrix(regularity);
    FullMatrix<double> B_new(regularity + 1);
    
    for (unsigned int i = 0; i <= regularity; ++i)
        for (unsigned int j = i; j <= regularity; ++j)
          {
            int mult1 = internal::factorial(j - i + 1);
            int mult2 = internal::factorial(j);
            
            B_new(j, i) = 0;
            double temp;
            
            for (unsigned int k = j - i + 1; k > 0; --k)
              {
                mult1 /= static_cast<int>(k);
                temp = mult2 * deriv_vals[k - 1] / mult1;
                B_new(j, i) += temp * B_std(j - k + 1, i);
              }
          }
      
    return B_new;
  }

  
  
  std::vector<double>
  HermiteCustomreg::hermite_poly_coeffs_customreg(const unsigned int degree,
                                                  const unsigned int regularity,
                                                  const unsigned int index)
  {
    Assert(
      degree < 14,
      ExcMessage(
        "Requested polynomial degree is too large, this may be due to the"
        " requested value being too large for numerical stability, or a"
        " bug in the code."));
    Assert(index <= degree,
           ExcMessage("The provided function index is out of range."));
    Assert(
      (2 * regularity + 1) < degree,
      ExcMessage(
        "Requested regularity is too high for the polynomial degree provided."));
    
    unsigned int          numnodes = degree - 2 * regularity - 1;
    std::vector<Point<1>> internal_nodes =
      create_chebyshevgausslobatto_nodes(numnodes);
    unsigned int local_index = index, group_index = 0;
    
    if (local_index > regularity)
      {
        local_index -= regularity + 1;
        ++group_index;
        
        if (local_index >= numnodes)
          {
            local_index -= numnodes;
            ++group_index;
          }
      }
      
    std::vector<double> poly_coeffs(degree + 1);
    double temp;
    
    if (group_index == 1)
      {
        temp                           = 1.0;
        const unsigned int edge_degree = regularity + 1;
        
        for (unsigned int i = 0; i <= edge_degree; ++i)
          {
            poly_coeffs[i + edge_degree] = temp;
            temp *= -static_cast<int>(edge_degree - i);
            temp /= static_cast<int>(i + 1);
          }
          
        unsigned int k = 1;
        
        for (unsigned int i = 0; i < numnodes; ++i)
          {
            if (i != local_index)
              {
                temp = -internal_nodes[i](0);
                
                for (unsigned int j = 2 * edge_degree + k; j > edge_degree; --j)
                  {
                    poly_coeffs[j] += poly_coeffs[j - 1];
                    poly_coeffs[j - 1] *= temp;
                  }
                  
                k++;
              }
          }
          
        temp               = poly_coeffs[degree];
        const double temp2 = internal_nodes[local_index](0);
        
        for (unsigned int i = degree; i > 0; --i)
          {
            temp *= temp2;
            temp += poly_coeffs[i - 1];
          }
          
        for (auto &cf : poly_coeffs)
          cf /= temp;
      }
    else
      {
        FullMatrix<double> B =
          modified_hermite_to_bernstein_matrix(regularity, internal_nodes);
        std::vector<double> coeff_sol(regularity + 1);
        double              fact = 1;
        
        for (unsigned int i = 0; i < local_index; ++i)
          {
            coeff_sol[i] = 0.0;
            fact *= i + 1;
          }
          
        coeff_sol[local_index] = fact / B(local_index, local_index);
        
        for (unsigned int i = local_index + 1; i <= regularity; ++i)
          {
            temp = 0;
            for (unsigned int j = local_index; j < i; ++j)
              temp -= B(i, j) * coeff_sol[j];
            coeff_sol[i] = temp / B(i, i);
          }
          
        if (group_index == 0)
          {
            temp = 1;
            
            for (unsigned int i = 0; i < regularity + 2; ++i)
              {
                for (unsigned int j = 0; j <= regularity; ++j)
                  poly_coeffs[i + j] += temp * coeff_sol[j];
                
                temp *= -static_cast<int>(regularity + 1 - i);
                temp /= static_cast<int>(i + 1);
              }
          }
        else
          {
            for (unsigned int i = 0; i <= regularity; ++i)
              {
                int sign = (local_index % 2 == 0) ? 1 : -1;
                int offset = regularity + 1;
                
                temp     = (numnodes % 2 == 0) ? coeff_sol[i] : -coeff_sol[i];
                temp *= sign;
                
                for (unsigned int j = 0; j <= i; ++j)
                  {
                    poly_coeffs[j + offset] += temp;
                    temp *= -static_cast<int>(i - j);
                    temp /= static_cast<int>(j + 1);
                  }
              }
          }
        for (unsigned int i = 0; i < numnodes; ++i)
          {
            temp = -internal_nodes[i](0);
            
            for (unsigned int j = 2 * regularity + i + 2; j > 0; --j)
              {
                poly_coeffs[j] += poly_coeffs[j - 1];
                poly_coeffs[j - 1] *= temp;
              }
          }
          
        temp = Utilities::pow(4, local_index);
        
        for (auto &it : poly_coeffs)
          it *= temp;
      }
      
    return poly_coeffs;
  }
  
  

  HermiteCustomreg::HermiteCustomreg(const unsigned int degree,
                                     const unsigned int regularity,
                                     const unsigned int index)
    : Polynomial<double>(
        hermite_poly_coeffs_customreg(degree, regularity, index))
    , degree(degree)
    , regularity(regularity)
  {
    unsigned int side_temp = 0, index_temp = index;
    
    if (index_temp > regularity)
      {
        ++side_temp;
        index_temp -= regularity + 1;
        
        if (index_temp > degree - 2 * regularity - 2)
          {
            ++side_temp;
            index_temp -= degree - 2 * regularity - 1;
          }
      }
      
    this->side       = side_temp;
    this->side_index = index_temp;
  }
  
  

  std::vector<Polynomial<double>>
  HermiteCustomreg::generate_complete_basis(const unsigned int degree,
                                            const unsigned int regularity)
  {
    Assert(
      degree < 14,
      ExcMessage(
        "Requested polynomial degree is too large, this may be due to the requested value being too large for numerical stability, or a bug in the code."));
    Assert(
      (2 * regularity + 1) < degree,
      ExcMessage(
        "Requested regularity is too high for the polynomial degree provided."));
    
    std::vector<Polynomial<double>> polys;
    
    for (unsigned int i = 0; i <= degree; ++i)
      polys.push_back(HermiteCustomreg(degree, regularity, i));
    
    return polys;
  }
} // namespace Polynomials

DEAL_II_NAMESPACE_CLOSE
