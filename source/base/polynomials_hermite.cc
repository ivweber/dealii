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

#include <deal.II/base/polynomials_hermite.h>

DEAL_II_NAMESPACE_OPEN

namespace Polynomials
{
  namespace
  {
    std::vector<double>
    hermite_poly_coeffs(const unsigned int regularity, const unsigned int index)
    {
      AssertIndexRange(index, 2 * regularity + 2);

      const unsigned int curr_index = index % (regularity + 1);
      const bool         side       = (index > regularity);

        // Protection needed here against underflow errors
      const int comparison_1 = static_cast<int>(regularity + 1 - curr_index);
      const int comparison_2 = side ? static_cast<int>(curr_index + 1) :
                                      static_cast<int>(regularity + 2);

      std::vector<double> poly_coeffs(2 * regularity + 2, 0.0);

      if (side) // g polynomials
        {
          int binomial_1 = (curr_index % 2) ? -1 : 1;

          for (int i = 0; i < comparison_2; ++i)
            {
              int inv_binomial = 1;

              for (int j = 0; j < comparison_1; ++j)
                {
                  int binomial_2 = 1;

                  for (int k = 0; k < j + 1; ++k)
                    {
                      poly_coeffs[regularity + i + k + 1] +=
                        binomial_1 * inv_binomial * binomial_2;
                      binomial_2 *= k - j;
                      binomial_2 /= k + 1;
                    }
                  inv_binomial *= regularity + j + 1;
                  inv_binomial /= j + 1;
                }
                 // Protection needed here against underflow errors
              binomial_1 *= -static_cast<int>(curr_index - i);
              binomial_1 /= i + 1;
            }
        }
      else // f polynomials
        {
          int binomial = 1;

          for (int i = 0; i < comparison_2; ++i)
            {
              int inv_binomial = 1;

              for (int j = 0; j < comparison_1; ++j)
                {
                  poly_coeffs[curr_index + i + j] += binomial * inv_binomial;
                  inv_binomial *= regularity + j + 1;
                  inv_binomial /= j + 1;
                }
               // Protection needed here against underflow errors
              binomial *= -static_cast<int>(regularity + 1 - i);
              binomial /= i + 1;
            }
        }

      // rescale coefficients by a factor of 4^curr_index to account for reduced
      // L2-norms
      double precond_factor = Utilities::pow(4, curr_index);
      for (auto &it : poly_coeffs)
        it *= precond_factor;

      return poly_coeffs;
    }
  } // namespace



  HermitePoly::HermitePoly(const unsigned int regularity,
                           const unsigned int index)
    : Polynomial<double>(hermite_poly_coeffs(regularity, index))
    , degree(2 * regularity + 1)
    , regularity(regularity)
    , side_index(index % (regularity + 1))
    , side((index > regularity) ? 1 : 0)
  {}



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
