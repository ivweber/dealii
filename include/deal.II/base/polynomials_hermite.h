/**
 * Header file for implementing higher regularity polynomials.
 * These can be either maximum regularity (degree 2q+1 for some
 * q) or have an intermediate regularity given by some r < (p-1)/2.
 * These will be implemented as separate classes for simplicity.
 */

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

namespace Polynomials
{
  /**
   * Class for Hermite polynomials enforcing the maximum posible level of
   * regularity in the FEM basis given the polynomial degree. This can only
   * result in odd degree methods, for other degrees HermiteCustomreg should be
   * used.
   *
   * Indices 0<=j<=q refer to polynomials corresponding to a non-zero derivative
   * of order j at x=0, and indices q+1<=j<=2q+1 refer to polynomials with a
   * non-zero derivative of order j-q-1 at x=1.
   */
  class HermiteMaxreg : public Polynomial<double>
  {
  public:
    /**
     * Constructor that takes an order of regularity to enforce up to (q),
     * and an index for which Hermite polynomial of the basis is to be
     * returned.
     */
    HermiteMaxreg(const unsigned int regularity, const unsigned int index)
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
    hermite_poly_coeffs_maxreg(const unsigned int regularity,
                               const unsigned int index);

  protected:
    int degree;
    int regularity;
    int side_index; // side_index=0 for left side, =1 for right side
    int side;
  };

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
