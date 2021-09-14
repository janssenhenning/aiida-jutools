# -*- coding: utf-8 -*-
###############################################################################
# Copyright (c), Forschungszentrum Jülich GmbH, IAS-1/PGI-1, Germany.         #
#                All rights reserved.                                         #
# This file is part of the aiida-jutools package.                             #
# (AiiDA JuDFT tools)                                                         #
#                                                                             #
# The code is hosted on GitHub at https://github.com/judftteam/aiida-jutools. #
# For further information on the license, see the LICENSE.txt file.           #
# For further information please visit http://judft.de/.                      #
#                                                                             #
###############################################################################
"""AiiDA JuTools.

For users:

We recommended to use this package with the import statement ``import aiida_jutools as jutools``. In your code,
you can then call all available tools like so: ``jutools.module.tool()``.

For developers:

- Place larger classes in separate modules, smaller functions in respective ``module/util.py``.
- Make all available at module level via import in ``module/__init__.py``. See existing files as templates.
- Also import each new module in the top-level ``__init__.py``.
"""
__version__ = "0.1.0-dev1"

# module-level imports
import code
import computer
import logging
import process_functions

