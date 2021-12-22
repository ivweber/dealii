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

  FullMatrix<double>
  HermitePoly::hermite_to_bernstein_matrix(const unsigned int regularity)
  {
    Assert(
      regularity < 8,
      ExcMessage(
        "The value passed to regularity is too high, this may be due to the requested value being too high for numerical stability, or due to a bug in the code."));
    
    const unsigned int sz = regularity + 1;
    std::vector<int>   coeffs(sz);
    FullMatrix<double> B(sz);
    int                mult;
    
    coeffs[0] = 1;
    for (unsigned int i = 1; i < sz; ++i)
      {
        mult      = -static_cast<int>(sz - i + 1);
        coeffs[i] = (mult * coeffs[i - 1]) / static_cast<int>(i);
      }

    for (unsigned int i = 0; i < sz; ++i)
      {
        B(i, i) = 1;
        for (unsigned int j = i + 1; j < sz; ++j)
          {
            B(i, j) = 0;
            B(j, i) = coeffs[j - i];
          }
      }
      
    return B;
  }
  
  

  std::vector<double>
  HermitePoly::hermite_poly_coeffs(const unsigned int regularity,
                                            const unsigned int index)
  {
    Assert(index < (2 * regularity + 2),
           ExcMessage("The provided function index is out of range."));
    
    FullMatrix<double> B            = hermite_to_bernstein_matrix(regularity);
    unsigned int       curr_index   = index % (regularity + 1);
    bool               at_next_node = (index > regularity);
    
    // Next node needs corrections based on g_k(x) = (-1)^{k} f_k(1-x)
    std::vector<double> bern_coeffs(regularity + 1, 0.0);
    
    bern_coeffs[curr_index] = 1.0;
    for (unsigned int i = curr_index + 1; i <= regularity; ++i)
      {
        double temp = 0;
        for (unsigned int j = 0; j < i; ++j)
          temp -= B(i, j) * bern_coeffs[j];
        bern_coeffs[i] = temp;
      }
      
    std::vector<double> poly_coeffs(2 * regularity + 2);
    double              curr_coeff;
    
    if (!at_next_node)
      {
        int                 temp_int = 1;
        std::vector<double> binom(regularity + 2);
        
        for (unsigned int i = 0; i < regularity + 2; ++i)
          {
            binom[i] = temp_int;
            temp_int *= -static_cast<int>(regularity + 1 - i);
            temp_int /= static_cast<int>(i + 1);
          }
          
        for (unsigned int i = 0; i <= regularity; ++i)
          for (unsigned int j = 0; j < regularity + 2; ++j)
            poly_coeffs[i + j] += bern_coeffs[i] * binom[j];
      }
    else
      {
        int sign = (curr_index % 2 == 0) ? 1 : -1;
        
        for (unsigned int i = 0; i <= regularity; ++i)
          {
            poly_coeffs[i] = 0.0;
            curr_coeff     = bern_coeffs[i] * sign;
            poly_coeffs[regularity + 1] += curr_coeff;
            int temp_int = 1;
            for (unsigned int j = 1; j <= i; ++j)
              {
                temp_int *= -static_cast<int>(i - j + 1);
                temp_int /= static_cast<int>(j);
                poly_coeffs[j + regularity + 1] += temp_int * curr_coeff;
              }
          }
      }
      
    // rescale coefficients by a factor of 4^curr_index to account for reduced
    // L2-norms
    double precond_factor = Utilities::pow(4, curr_index);
    for (auto &it : poly_coeffs)
      it *= precond_factor;
    
    return poly_coeffs;
  }
  
  

  std::vector<Polynomial<double>>
  HermitePoly::generate_complete_basis(const unsigned int regularity)
  {
    std::vector<Polynomial<double>> polys;
    const unsigned int              sz = 2 * regularity + 2;
    
    for (unsigned int i = 0; i < sz; ++i)
      polys.push_back(HermitePoly(regularity, i));
    
    return polys;
  }
} // namespace Polynomials

DEAL_II_NAMESPACE_CLOSE
