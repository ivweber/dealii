// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2023 by the deal.II authors
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

#ifndef dealii_cut_hermite_tools_h
#define dealii_cut_hermite_tools_h

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <map>
#include <vector>

DEALII_NAMESPACE_OPEN

namespace CutHermiteTools
{
    /**
     * @name Additional functions to make using cutFEM with Hermite
     * elements simpler for the end user.
     * @{
     */

    

    /** @} */
} //CutHermiteTools

DEALII_NAMESPACE_CLOSE

#endif