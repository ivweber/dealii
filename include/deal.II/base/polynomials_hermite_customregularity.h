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



#ifndef dealii_hermite_polynomials_customregularity
#define dealii_hermite_polynomials_customregularity

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
 * Header file for implementing higher regularity polynomials that
 * do not automatically impose the maximum possible regularity. If this
 * regularity is desired then the HermitePoly class should be used instead
 * for efficiency. These polynomials have an intermediate regularity given 
 * by some r < (p-1)/2.
 */

namespace Polynomials
{
  /**
   * Class for Hermite polynomials of degree p enforcing an intermediate level
   * of regularity q. This has a strict requirement that 2q+1 < p, since with
   * equality the maximum regularity class should be used and higher than the
   * maximum regularity is not possible. The additional polynomial degree is
   * accounted for by using Lagrange-style interpolation nodes on the interior
   * of the element, positioned at Chebyshev Gauss-Lobatto points.
   *
   * Indices 0 <= j <= q refer to polynomials with a non-zero derivative of
   * order j at x=0 and zero values at the intermediate nodal points, indices
   * q+1 <= j <= p-q-1 refer to polynomials with derivatives up order q equal to
   * 0 at both x=0 and x=1 and a non-zero value at intermediate node with index
   * j-q (node index 0 refers to x=0, node index p-2q refers to x=1), and
   * indices p-q <= j <= p refer to polynomials with a non-zero derivative of
   * order j-p+q at x=1 and zero value at the intermediate nodes.
   */
  class HermiteCustomreg : public Polynomial<double>
  {
  public:
    /**
     * Constructor that takes a polynomial degree p for the basis and an
     * order of regularity q to enforce and an index for which polynomial to
     * return.
     */
    HermiteCustomreg(const unsigned int degree,
                     const unsigned int regularity,
                     const unsigned int index);

    /**
     * Function that returns a complete basis of the specified regularity
     * polynomials in a vector.
     */
    static std::vector<Polynomial<double>>
    generate_complete_basis(const unsigned int degree,
                            const unsigned int regularity);

    /**
     * Create the interpolation nodes to complete the basis to the required
     * degree
     */
    static std::vector<Point<1>>
    create_chebyshevgausslobatto_nodes(const unsigned int no_nodes);

    /**
     * Helper function for generating a vector of derivatives for the polynomial
     * interpolating the internal nodes of the basis
     */
    static std::vector<double>
    interpolant_derivatives(const std::vector<Point<1>> &nodes);

    /**
     * Define a matrix similar to the hermite_to_bernstein matrix, but for
     *derivatives of P(x) z_{k,q+1}(x), where P is the polynomial with roots at
     *the internal CGL points and z_{k,q+1}(x) = x^{k} (1-x)^{q+1}.
     */
    static FullMatrix<double>
    modified_hermite_to_bernstein_matrix(const unsigned int regularity,
                                         const std::vector<Point<1>> &nodes);

    /**
     * Function to obtain the polynomial expansion of a given polynomial
     */
    static std::vector<double>
    hermite_poly_coeffs_customreg(const unsigned int degree,
                                  const unsigned int regularity,
                                  const unsigned int index);

  protected:
    int degree;
    int regularity;
    int side_index; // side=0 for left side, =1 for nodal function, =2 for right
                    // side
    int side;
  };
} // namespace Polynomials

DEAL_II_NAMESPACE_CLOSE

#endif
