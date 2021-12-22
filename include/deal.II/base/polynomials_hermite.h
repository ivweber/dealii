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



#ifndef dealii_hermite_polynomials
#define dealii_hermite_polynomials

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/full_matrix.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <vector>



DEAL_II_NAMESPACE_OPEN

/**
 * Header file for implementing higher regularity polynomials.
 * These always have an odd-numbered polynomial degree and
 * impose the maximum regularity possible (continuity up to the 
 * r-th derivative, for polynomial degree 2r+1).
 */

namespace Polynomials
{
  /**
   * Class for Hermite polynomials enforcing the maximum posible level of
   * regularity in the FEM basis given the polynomial degree. This can only
   * result in odd degree methods.
   *
   * Indices 0<=j<=q refer to polynomials corresponding to a non-zero derivative
   * of order j at x=0, and indices q+1<=j<=2q+1 refer to polynomials with a
   * non-zero derivative of order j-q-1 at x=1.
   */
  class HermitePoly : public Polynomial<double>
  {
  public:
    /**
     * Constructor that takes an order of regularity to enforce up to (q),
     * and an index for which Hermite polynomial of the basis is to be
     * returned.
     */
    HermitePoly(const unsigned int regularity, const unsigned int index)
      : Polynomial<double>(hermite_poly_coeffs_maxreg(regularity, index))
      , degree(2 * regularity + 1)
      , regularity(regularity)
      , side_index(index % (regularity + 1))
      , side((index > regularity) ? 1 : 0){};

    /**
     * Function that returns a complete basis of maximum regularity Hermite
     * polynomials of order 2q+1 as a vector.
     */
    static std::vector<Polynomial<double>>
    generate_complete_basis(const unsigned int regularity);

    /**
     * Function that returns a matrix B containing derivatives of Bernstein
     * polynomials, where (j! B_jk) is the j-th derivative of polynomial
     * x^k (1-x)^(q+1) at x=0. This is useful both for computing the basis
     * and assigning weights to local degrees of freedom at hanging nodes. The
     * rescaling is to improve performance of solution.
     *
     * All entries are known to be integers, so the matrix stores ints to
     * reduce floating point errors. Care should be taken when solving as a
     * result.
     */
    static FullMatrix<double>
    hermite_to_bernstein_matrix(const unsigned int regularity);
    
    /**
     * Function that returns the coefficients for a polynomial expansion of
     * a given Hermite polynomial. This can be passed to the polynomial
     * constructor to immediately obtain a Hermite polynomial of that object
     * type.
     */
    static std::vector<double>
    hermite_poly_coeffs(const unsigned int regularity,
                        const unsigned int index);

  protected:
    int degree;
    int regularity;
    int side_index; // side_index=0 for left side, =1 for right side
    int side;
  };
} // namespace Polynomials

DEAL_II_NAMESPACE_CLOSE

#endif
