// ---------------------------------------------------------------------
//
// Copyright (C) 2006 - 2020 by the deal.II authors
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


#include <deal.II/base/data_out_base.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <string>
#include <vector>

#include "../tests.h"

#include "patches.h"

// Output data on repetitions of the unit hypercube

// define this as 1 to get output into a separate file for each testcase
#define SEPARATE_FILES 0


template <int dim, int spacedim>
void
check(DataOutBase::PovrayFlags flags, std::ostream &out)
{
  const unsigned int np = 4;

  std::vector<DataOutBase::Patch<dim, spacedim>> patches(np);

  create_patches(patches);

  std::vector<std::string> names(5);
  names[0] = "x1";
  names[1] = "x2";
  names[2] = "x3";
  names[3] = "x4";
  names[4] = "i";
  std::vector<
    std::tuple<unsigned int,
               unsigned int,
               std::string,
               DataComponentInterpretation::DataComponentInterpretation>>
    vectors;
  DataOutBase::write_povray(patches, names, vectors, flags, out);
}


template <int dim>
void
check_cont(unsigned int             ncells,
           unsigned int             nsub,
           DataOutBase::PovrayFlags flags,
           std::ostream            &out)
{
  std::vector<DataOutBase::Patch<dim, dim>> patches;

  create_continuous_patches(patches, ncells, nsub);

  std::vector<std::string> names(1);
  names[0] = "CutOff";
  std::vector<
    std::tuple<unsigned int,
               unsigned int,
               std::string,
               DataComponentInterpretation::DataComponentInterpretation>>
    vectors;
  DataOutBase::write_povray(patches, names, vectors, flags, out);
}


template <int dim, int spacedim>
void
check_all(std::ostream &log)
{
#if SEPARATE_FILES == 0
  std::ostream &out = log;
#endif

  char name[100];
  //  const char* format = "%d%d%d%s.pov";
  DataOutBase::PovrayFlags flags;

  if (true)
    {
      sprintf(name, "cont%d%d%d.pov", dim, 4, 4);
#if SEPARATE_FILES == 1
      std::ofstream out(name);
#else
      out << "==============================\n"
          << name << "\n==============================\n";
#endif
      check_cont<dim>(4, 4, flags, out);
    }

  flags.external_data = true;
  if (true)
    {
      sprintf(name, "cont%d%d%dtri.pov", dim, 4, 4);
#if SEPARATE_FILES == 1
      std::ofstream out(name);
#else
      out << "==============================\n"
          << name << "\n==============================\n";
#endif
      check_cont<dim>(4, 4, flags, out);
    }

  flags.smooth = true;
  if (true)
    {
      sprintf(name, "cont%d%d%dsmooth.pov", dim, 4, 4);
#if SEPARATE_FILES == 1
      std::ofstream out(name);
#else
      out << "==============================\n"
          << name << "\n==============================\n";
#endif
      check_cont<dim>(4, 4, flags, out);
    }

  flags.bicubic_patch = true;
  if (true)
    {
      sprintf(name, "cont%d%d%dbic.pov", dim, 4, 3);
#if SEPARATE_FILES == 1
      std::ofstream out(name);
#else
      out << "==============================\n"
          << name << "\n==============================\n";
#endif
      check_cont<dim>(4, 3, flags, out);
    }
}

int
main()
{
  std::ofstream logfile("output");
  //  check_all<1,1>(logfile);
  //  check_all<1,2>(logfile);
  check_all<2, 2>(logfile);
  check_all<2, 3>(logfile);
  //  check_all<3,3>(logfile);
}
