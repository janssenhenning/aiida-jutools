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
"""Tools for working with aiida-kkr nodes."""

from typing import Union as _Union, Dict as _Dict, Any as _Any
from collections import OrderedDict as _OrderedDict
import enum as _enum
import os as _os
import datetime as _datetime
import pytz as _pytz

import numpy as _np
import pandas as _pd
import aiida.orm
from aiida.orm import CalcJobNode as _CalcJobNode, WorkChainNode as _WorkChainNode
from aiida.orm import QueryBuilder as _QueryBuilder, Group as _Group, RemoteData as _RemoteData, \
    StructureData as _StructureData
from aiida_kkr.calculations import VoronoiCalculation as _VoronoiCalculation
from aiida_kkr.workflows import kkr_imp_wc as _kkr_imp_wc, kkr_scf_wc as _kkr_scf_wc, \
    kkr_startpot_wc as _kkr_startpot_wc

from masci_tools.util import math_util as _math_util
from masci_tools.util.chemical_elements import ChemicalElements as _ChemicalElements
from masci_tools.util.constants import ANG_BOHR_KKR as _ANG_BOHR_KKR
from masci_tools.util import python_util as _python_util


def check_if_kkr_calc_converged(kkr_calc: _CalcJobNode):
    """Assert (fail if false) that kkr calculation has converged.

    DEVNOTE: used aiida base node type for argument type so it works with all kkr calc node types.

    E.g. needed for host GF writeout

    Reference: https://aiida-kkr.readthedocs.io/en/stable/user_guide/calculations.html#special-run-modes-host-gf-writeout-for-kkrimp

    :param kkr_calc: performed kkr calculation
    """
    try:
        assert kkr_calc.outputs.output_parameters.get_dict()["convergence_group"]["calculation_converged"] == True
    except KeyError as err:
        print("Error: calculation is not a kkr calculation.")
        raise err


def query_kkr_wc(cls=_kkr_imp_wc, symbols: list = ['H', 'H'], group=None) -> _QueryBuilder:
    """Query kkr workchains based on their input structures.

    Constraints:

    - if kkr_scf_wc, and symbols given, queries with first symbol only (elemental crystal).
    - if kkr_imp_wc, requires symbols, queries with first symbol = impurity, second symbol = host crystal.

    For general workchain queries, use :py:func:`~aiida_jutools.util_process.query_processes` instead.

    :param cls: kkr workchain class
    :type cls: kkr_scf_wc or kkr_imp_wc
    :param group: given: search in group, not: search in database
    :type group: aiido.orm.Group
    :param symbols: list of chemical element symbols.
    :type symbols: list of str
    :return: the built query for matching workchains
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    qb = _QueryBuilder()
    if group:
        qb.append(_Group, filters={'label': group.label}, tag='group')
    if issubclass(cls, _kkr_scf_wc):
        if group:
            qb.append(_kkr_scf_wc, with_group='group', tag='workchain', project='*')
        else:
            qb.append(_kkr_scf_wc, tag='workchain', project='*')
        if symbols:
            qb.append(_StructureData, with_outgoing='workchain',
                      filters={'attributes.kinds.0.name': symbols[0]})
            # # alternative: require extras
            # qb.append(_StructureData, with_outgoing='workchain', filters={"extras.symbol": symbols[0]})
    elif issubclass(cls, _kkr_imp_wc):
        if not symbols:
            raise KeyError("No symbols supplied.")
        if len(symbols) == 2:
            elmts = _ChemicalElements()
            imp_number = elmts[symbols[0]]
            # wc.inputs.impurity_info.attributes['Zimp']
            if group:
                qb.append(_kkr_imp_wc, with_group='group', tag='imp_wc', project='*')
            else:
                qb.append(_kkr_imp_wc, tag='imp_wc', project='*')
            qb.append(aiida.orm.Dict, with_outgoing='imp_wc', filters={'attributes.Zimp': imp_number})
            qb.append(_RemoteData, with_outgoing='imp_wc', tag='remotedata')
            qb.append(_kkr_scf_wc, with_outgoing='remotedata', tag='scf_wc')
            qb.append(_StructureData, with_outgoing='scf_wc',
                      filters={'attributes.kinds.0.name': symbols[1]})
            # # alternative: require extras
            # qb.append(_StructureData, with_outgoing='scf_wc', filters={"extras.symbol": symbols[1]})

            # # alternative: require extras
            # # note: don't set symbol in workchain extras anymore, so this is deprecated.
            # imp_symbol = ":".join(symbols)
            # if group:
            #     qb.append(_kkr_imp_wc, with_group='group', filters={"extras.embedding_symbol": imp_symbol})
            # else:
            #     qb.append(_kkr_imp_wc, filters={"extras.embedding_symbol": imp_symbol})
        else:
            raise NotImplementedError(f"query not implemented for kkr_imp_wc with no. of symbols other than 2.")
    else:
        raise NotImplementedError(f"workchain query not implemented for class {cls}.")
    return qb  # .all(flat=True)


def query_structure_from(wc: _WorkChainNode) -> _StructureData:
    """Get structure from kkr workchain.

    :param wc: workchain
    :type wc: WorkChainNode of subtype kkr_scf_wc or kkr_imp_wc
    :return: structure if found else None
    :rtype: StructureData
    """
    from aiida.orm import WorkChainNode
    from aiida.engine import WorkChain
    assert isinstance(wc, WorkChain) or isinstance(wc, WorkChainNode)

    wc_cls_str = wc.attributes['process_label']
    if wc_cls_str == 'kkr_scf_wc':
        # solution1: timing 7ms
        return wc.inputs.structure
        # # solution2: timing 27ms
        # return VoronoiCalculation.find_parent_structure(wc)
    elif wc_cls_str == 'kkr_imp_wc':
        # solution1: timing 18 ms
        qb = _QueryBuilder()
        qb.append(_StructureData, tag='struc', project='*')
        qb.append(_kkr_scf_wc, with_incoming='struc', tag='scf_wc')
        qb.append(_RemoteData, with_incoming='scf_wc', tag='remotedata')
        qb.append(_kkr_imp_wc, with_incoming='remotedata', filters={'uuid': wc.uuid})
        res = qb.all(flat=True)
        return res[0] if res else None

        # # solution2: timing 23ms
        # scf = wci.inputs.remote_data_host.get_incoming(node_class=_kkr_scf_wc).all_nodes()
        # return scf[0].inputs.structure if scf else None
    else:
        raise NotImplementedError(f"workchain query not implemented for class {wc_cls_str}.")


def find_Rcut(structure: _StructureData, shell_count: int = 2, rcut_init: float = 7.0) -> float:
    """For GF writeout / impurity workflows: find radius such that only nearest-neighbor shells are included.

    :param structure: structure.
    :param shell_count: include this many nearest-neighbor shells around intended impurity site.
    :param rcut_init: initial maximal rcut value, will be iteratively decreased until fit.
    :return: rcut radius
    """
    import numpy as np

    struc_pmg = structure.get_pymatgen()

    rcut = rcut_init
    nc = 0
    while nc < shell_count:
        dists = struc_pmg.get_neighbor_list(rcut, sites=[struc_pmg.sites[0]])[-1]
        dists = [np.round(i, 5) for i in dists]
        dists.sort()
        nc = len(set(dists))
        rcut += 5

    if nc > shell_count:
        n3start = dists.index(np.sort(list(set(dists)))[shell_count])
        d0, d1 = dists[n3start - 1:n3start + 1]
        rcut = d0 + (d1 - d0) / 2.

    return rcut


class KkrConstantsVersion(_enum.Enum):
    """Enum for labeling different KKR constants version.

    Used by :py:class:`~aiida_jutools.util_kkr.KkrConstantsChecker`.

    The enum values represent the respective constants values from different time spans

    - :py:func:`~masci_tools.io.common_functions.get_Ang2aBohr`
    - :py:func:`~masci_tools.io.common_functions.get_aBohr2Ang`
    - :py:func:`~masci_tools.io.common_functions.get_Ry2eV`
    - :py:data:`~masci_tools.util.constants.ANG_BOHR_KKR`
    - :py:data:`~masci_tools.util.constants.RY_TO_EV_KKR`

    Here is an overview of their values from the commit history timespan when the values underwent change.

    ==========  ===========  =====  ===================  ====================  ==================
    date        commit hash  type   ang 2 bohr constant  bohr to ang constant  ry to ev constant
    ==========  ===========  =====  ===================  ====================  ==================
    2018-10-26  04d55ea             1.8897261254578281   0.5291772106700000    13.605693009000000
    2021-02-16  c171563             1.8897261249935897   0.5291772108000000    13.605693122994000
    2021-04-28  66953f8      'old'  1.8897261254578281   0.5291772106700000    13.605693009000000
    2021-04-28  66953f8      'new'  1.8897261246257702   0.5291772109030000    13.605693122994000
    ==========  ===========  =====  ===================  ====================  ==================

    Use :py:attr:`~aiida_jutools.util_kkr.KkrConstantsVersion.OLD.description` (or on any other enum) to get a
    machine-readable version of this table.

    So we have the following correspondence for ang 2 bohr constant / bohr to ang constant:

    - OLD: [2018-10-26, 2021-02-16] and [2021-04-28,] (type 'old')
    - INTERIM: [2021-02-16, 2021-04-28]
    - NEW: [2021-04-28,] (type 'new')

    For ry to ev constant, we have:

    - OLD: as above
    - INTERIM: same as NEW.
    - NEW: [2021-02-16, 2021-04-28] and [2021-04-28,] (type 'new')

    For constants values reverse-calculated from finished workchain for classification, we have the additional enums:

    - NEITHER: for constants values recalculated from workchains which fit neither of the above by a wide margin
    - UNDECISIVE: for constants values recalculated from workchains which fit neither of the above but are in range

    Note: The order here reflects the importance. NEW should be preferred, OLD be used for old workchains performed
    with these values, INTERIM should be avoided since masci-tools versions 2021-04-28 do not know these values,
    NEITHER and UNDECISIVE are for workchain, not constants, reverse classification purposes only.
    """
    NEW = 0
    OLD = 1
    INTERIM = 2
    NEITHER = 3
    UNDECISIVE = 4

    @property
    def description(self) -> _Union[_Dict[str, _Union[str, _datetime.datetime]], str]:
        """Describe constants versions.

        Returns either a dictionary or a string, depending on the enum.

        The returned dictionary describes from when to when the respective KKR constants
        version was defined in the respective masci-tools version, here denoted by
        commit hashes, as they are more accurate than the librarie's version numbers.
        The left and right time limits are denoted by datetime objects of year one, and now.

        This can be taken as a machine-readable indicator for comparison against classification results of
        :py:class:`~aiida_jutools.util_kkr.KkrConstantsChecker` to validate whether the constants versions
        there found for a workchain fit within the respective constants version's timefame described here.

        Keep in mind though that the respective inspected workchain might have been run with an older masci-tools
        version at creation time. So any constants version older than the workchain's creation time are legit,
        only newer ones are impossible.
        """
        if self.name == KkrConstantsVersion.NEW.name:
            return {'commit': "66953f8",
                    'valid_from': _datetime.datetime(year=2021, month=4, day=28,
                                                     hour=14, minute=2, second=0,
                                                     microsecond=0, tzinfo=_pytz.UTC),
                    'valid_until': _python_util.now()
                    }
        elif self.name == KkrConstantsVersion.INTERIM.name:
            return {'commit': "c171563",
                    'valid_from': _datetime.datetime(year=2021, month=2, day=16,
                                                     hour=19, minute=40, second=0,
                                                     microsecond=0, tzinfo=_pytz.UTC),
                    'valid_until': _datetime.datetime(year=2021, month=4, day=28,
                                                      hour=14, minute=2, second=0,
                                                      microsecond=0, tzinfo=_pytz.UTC)
                    }
        elif self.name == KkrConstantsVersion.OLD.name:
            return {'commit': "04d55ea",
                    'valid_from': _datetime.datetime(year=1, month=1, day=1,
                                                     hour=0, minute=0, second=0,
                                                     microsecond=0, tzinfo=_pytz.UTC),
                    'valid_until': _datetime.datetime(year=2021, month=2, day=16,
                                                      hour=19, minute=40, second=0,
                                                      microsecond=0, tzinfo=_pytz.UTC)
                    }
        elif self.name in [KkrConstantsVersion.NEITHER.name, KkrConstantsVersion.UNDECISIVE.name]:
            return f"For classification of aiida-kkr workchains by class {KkrConstantsVersionChecker.__name__}."
        else:
            raise NotImplementedError("Enum with undefined behavior. Contact developer.")


class KkrConstantsVersionChecker:
    """Find out with which version of constants ``ANG_BOHR_KKR``, ``RY_TO_EV_KKR`` finished aiida-kkr workchain were run.

    Between 2021-02-16 and 2021-04-28, the the values of the conversion constants ``ANG_BOHR_KKR`` and
    ``RY_TO_EV_KKR`` in :py:mod:`~masci_tools.util.constants` were changed from previous values to a set of
    intermediate values, and then finally to NIST values. See :py:class:`~aiida_jutools.util_kkr.KkrConstantsVersion`
    docstring for a complete list. The ``constants`` module mentioned above offers an option to switch
    back to the older constants versions, see its documentation.

    As a result, calculations with the old constants versions cannot be reused in new calculations, otherwise
    the calculation fails with the error ``[read_potential] error ALAT value in potential is does not match``
    in the ``_scheduler-stderr.txt`` output.

    This class checks aiida-kkr workchains for the constants version it likely was performed with. The result
    is a DataFrame with one row for each checked workchain.

    Currently, this class only reverse-calculates the ``ANG_BOHR_KKR`` constant from a workchain for version checking.
    The ``RY_TO_EV_KKR`` constant is not used.
    """

    def __init__(self):
        """Initialization reads the constants' runtime versions and cross-checks with environment settings."""

        # problem is that masci_tools.util.constants constants ANG_BOHR_KKR, RY_TO_EV_KKR definitions (values) depend on
        # the value of the env var os.environ['MASCI_TOOLS_USE_OLD_CONSTANTS'] at module initialization. After that,
        # the definition stays fixed, and changing the value of the env var does not alter it anymore.
        # So, only way to have access to all versions at runtime is to redefine the constants values here
        # (using the same values as there).
        # This also means that we could just query the current constants value to decide whether current env loaded
        # the old or new values. But we will still check the env var in order to cross-check findings. If they don't
        # agree, the implementation logic has likely changed, and this code may be out of order.

        #######################
        # 1) init internal data structures
        self._ANG_BOHR_KKR = {  # order importance (not by value): NEW > OLD > INTERIM
            KkrConstantsVersion.NEW: 1.8897261246257702,
            KkrConstantsVersion.INTERIM: 1.8897261249935897,
            KkrConstantsVersion.OLD: 1.8897261254578281,
        }
        self._RY_TO_EV_KKR = {  # order importance (not by value): NEW > OLD
            KkrConstantsVersion.NEW: 13.605693122994,
            KkrConstantsVersion.INTERIM: 13.605693122994,
            KkrConstantsVersion.OLD: 13.605693009,
        }
        self._runtime_const_type = {
            'ANG_BOHR_KKR': None,
            'RY_TO_EV_KKR': None
        }
        # create an empty DataFrame to hold one row of data for each check workchain.
        self._df_index_name = 'workchain_uuid'
        self._df_schema = {
            'ctime': object,  # workchain ctime
            'group': str,  # group label, if specified
            'ANG_BOHR_KKR': _np.float64,  # recalculated from alat, bravais
            'constants_version': object,  # type of recalculated ANG_BOHR_KKR (old, new, neither) based on abs_tol
            'diff_new': _np.float64,  # abs. difference recalculated - new ANG_BOHR_KKR value
            'diff_old': _np.float64,  # abs. difference recalculated - old ANG_BOHR_KKR value
            'diff_interim': _np.float64,  # abs. difference recalculated - interim ANG_BOHR_KKR value
        }
        self._df = _pd.DataFrame(columns=self._df_schema.keys()).astype(self._df_schema)
        self._df.index.name = self._df_index_name

        #######################
        # 2) read in current constants values and cross-check with environment

        # get the current ANG_BOHR_KKR value
        # note: aiida-kkr uses masci_tools.io.common_functions get_Ang2aBohr (=ANG_BOHR_KKR),
        #       get_aBohr2Ang() (=1/ANG_BOHR_KKR), get_Ry2eV (=RY_TO_EV_KKR) instead, but this is redundant.
        #       Here we import the constants directly.
        # note:
        msg_suffix = "This could indicate an implementation change. " \
                     "As a result, this function might not work correctly anymore."

        if _ANG_BOHR_KKR == self.ANG_BOHR_KKR[KkrConstantsVersion.NEW]:
            self._runtime_const_type['ANG_BOHR_KKR'] = KkrConstantsVersion.NEW
        elif _ANG_BOHR_KKR == self.ANG_BOHR_KKR[KkrConstantsVersion.INTERIM]:
            self._runtime_const_type['ANG_BOHR_KKR'] = KkrConstantsVersion.INTERIM
        elif _ANG_BOHR_KKR == self.ANG_BOHR_KKR[KkrConstantsVersion.OLD]:
            self._runtime_const_type['ANG_BOHR_KKR'] = KkrConstantsVersion.OLD
        else:
            self._runtime_const_type['ANG_BOHR_KKR'] = KkrConstantsVersion.NEITHER
            print(f"Warning: The runtime value of constant ANG_BOHR_KKR matches no expected value. {msg_suffix}")

        # env var cases: 4: None, 'Interim', 'True', not {None, 'True', 'Interim'}.
        # const type cases: 4: NEW, OLD, INTERIM, NEITHER.
        # cross-product: 4 x 4 = 16.
        # this assumes that current masci-tools version supports the environment switch WITH the 'Interim' option,
        # i.e. from 2021-01-08 or newer (switch was implemented 2021-04-28, without 'Interim' option).
        #
        # | env var                       | const type | valid | defined | reaction  | case |
        # | ------------------            | ---------- | ----- | ------- | --------- | ---- |
        # | None                          | NEW        | yes   | yes     | pass      | E    |
        # | None                          | INTERIM    | no    | no      | exception | A    |
        # | None                          | OLD        | no    | no      | exception | A    |
        # | None                          | NEITHER    | no    | no      | exception | A    |
        # | 'Interim                      | New        | no    | no      | exception | B    |
        # | 'Interim'                     | INTERIM    | yes   | yes     | pass      | E    |
        # | 'Interim'                     | OLD        | yes   | yes     | exception | B    |
        # | 'Interim'                     | NEITHER    | no    | no      | exception | B    |
        # | 'True'                        | New        | no    | no      | exception | C    |
        # | 'True'                        | INTERIM    | no    | no      | exception | C    |
        # | 'True'                        | OLD        | yes   | yes     | pass      | E    |
        # | 'True'                        | NEITHER    | no    | no      | exception | C    |
        # | not {None, 'True', 'Interim'} | NEW        | no    | no      | pass(1)   | E    |
        # | not {None, 'True', 'Interim'} | INTERIM    | no    | no      | exception | D    |
        # | not {None, 'True', 'Interim'} | OLD        | no    | no      | exception | D    |
        # | not {None, 'True', 'Interim'} | NEITHER    | no    | no      | exception | D    |
        # Annotations:
        # - case 'D' = 'else' = 'pass'.
        # - (1): passes with warning, from const type NEITHER above.

        # double-check with environment variable
        env_var_key = 'MASCI_TOOLS_USE_OLD_CONSTANTS'
        env_var_val = _os.environ.get(env_var_key, None)
        const_type = self._runtime_const_type['ANG_BOHR_KKR']

        cases = {
            'A': (const_type != KkrConstantsVersion.NEW and env_var_val is None),
            'B': (const_type != KkrConstantsVersion.INTERIM and env_var_val == 'Interim'),
            'C': (const_type != KkrConstantsVersion.OLD and env_var_val == 'True'),
            'D': (const_type != KkrConstantsVersion.NEW and env_var_val not in [None, 'True', 'Interim'])
        }
        if cases['A'] or cases['D']:
            raise ValueError(
                f"Based on environment variable {env_var_key}={env_var_val}, I expected constant values to "
                f"be of type {KkrConstantsVersion.NEW}, but they are of type {const_type}. "
                f"{msg_suffix}")
        elif cases['B']:
            raise ValueError(
                f"Based on environment variable {env_var_key}={env_var_val}, I expected constant values to "
                f"be of type {KkrConstantsVersion.INTERIM}, but they are of type {const_type}. "
                f"{msg_suffix}")
        elif cases['C']:
            raise ValueError(
                f"Based on environment variable {env_var_key}={env_var_val}, I expected constant values to "
                f"be of type {KkrConstantsVersion.OLD}, but they are of type {const_type}. "
                f"{msg_suffix}")
        else:
            pass

    @property
    def ANG_BOHR_KKR(self) -> dict:
        """All constants versions of the conversion constant ``ANG_BOHR_KKR`` (Angstrom to Bohr radius)."""
        return self._ANG_BOHR_KKR

    @property
    def RY_TO_EV_KKR(self) -> dict:
        """All constants versions of the conversion constant ``RY_TO_EV_KKR`` (Rydberg to electron Volt)."""
        return self._RY_TO_EV_KKR

    def get_runtime_type(self, constant_name: str = 'ANG_BOHR_KKR') -> KkrConstantsVersion:
        """Get KKR constant type (old, new or neither) of runtime constant value.

        :param constant_name: name of the constant.
        """
        if constant_name in self._runtime_const_type:
            return self._runtime_const_type[constant_name]
        else:
            print(f"Warning: Unknown constant name '{constant_name}'. "
                  f"Known constant names: {self._runtime_const_type[constant_name]}. "
                  f"I will return nothing.")

    @property
    def df(self) -> _pd.DataFrame:
        """DataFrame containing all checked workchain data."""
        return self._df

    def clear(self):
        """Clear all previous workchain checks."""
        # drop all rows from dataframe
        self._df = self._df.drop(labels=self._df.index)

    def check_single_workchain(self, wc: _WorkChainNode,
                               set_extra: bool = False,
                               zero_threshold: float = 1e-15,
                               group_label: str = None):
        """Classify a finished workchain by its used KKR constants version by reverse-calculation.

        The result is available as a dataframe :py:attr:`~aiida_jutools.util_kkr.KkrConstantsVersionChecker.df`.

        Note: Will always check constants version by recalculating it, even if it may have already been set as an
        extra. To classify with the extra instead, use method TODO.

        :param wc: finished aiida-kkr workchain. Must have a ``kkr_startpot_wc`` descendant.
        :param zero_threshold: Set structure cell elements below this threshold to zero to counter rounding errors.
        :param set_extra: True: Set an extra on the workchain denoting the identified KKR constants version and values.
        :param group_label: optional: specify group label the workchain belongs to.
        """

        if wc.uuid in self._df.index:
            print(f"Info: skipping Workchain {wc}: is already checked.")
            return

        #######################
        # 1) init internal variables

        # temp values for new dataframe row elements
        row = {key: _np.NAN for key in self._df_schema}
        ALATBASIS = None
        BRAVAIS = None
        POSITIONS = None
        ANG_BOHR_KKR = None
        constants_version = None

        structure = query_structure_from(wc)

        structure_cell = _np.array(structure.cell)
        _math_util.set_zero_below_threshold(structure_cell, threshold=zero_threshold)

        structure_positions = []
        for sites in structure.sites:
            structure_positions.append(sites.position)
        structure_positions = _np.array(structure_positions)
        _math_util.set_zero_below_threshold(structure_positions, threshold=zero_threshold)

        #######################
        # 2) Read original alat and bravais from first inputcard
        # For now, this is implemented for aiida-kkr workchains with a single
        #  kkr_startpot_wc > VoronoiCalculation descendant only.

        startpots = wc.get_outgoing(node_class=_kkr_startpot_wc).all_nodes()

        msg_prefix = f"Warning: skipping Workchain {wc}"
        msg_suffix = f"Method not implemented for such workchains"
        if not startpots:
            print(f"{msg_prefix}: Does not have a {_kkr_startpot_wc.__name__} descendant. {msg_suffix}.")
            return
        else:
            vorocalcs = None
            # workchain might have several startpot descendants, one of which should hava a vorocalc descendant.
            for startpot in startpots:
                vorocalcs = startpot.get_outgoing(node_class=_VoronoiCalculation).all_nodes()
                if vorocalcs:
                    break
            if not vorocalcs:
                print(f"{msg_prefix}: Does not have a {_VoronoiCalculation.__name__} descendant. {msg_suffix}.")
                return
            else:
                vorocalc = vorocalcs[0]
                # vorocalc.get_retrieve_list()
                try:
                    inputcard = vorocalc.get_object_content('inputcard')
                    inputcard = inputcard.split('\n')

                    # read alat value
                    indices = [idx for idx, line in enumerate(inputcard) if 'ALATBASIS' in line]
                    if len(indices) == 1:
                        ALATBASIS = float(inputcard[indices[0]].split()[1])
                    else:
                        print(f"{msg_prefix}: Could not read 'ALATBASIS' value from inputcard file. {msg_suffix}.")
                        return

                    def read_field(keyword: str):
                        lines = []
                        reading = False
                        for i, line in enumerate(inputcard):
                            if reading:
                                if line.startswith(' '):
                                    lines.append(line)
                                else:
                                    reading = False
                            if keyword in line:
                                reading = True
                        array = []
                        for line in lines:
                            array.append([float(numstr) for numstr in line.split()])
                        array = _np.array(array)
                        return array

                    # read bravais value(s)
                    # Typically, inputcard has line 'BRAVAIS', followed by 3 linex of 1x3 bravais matrix values.
                    BRAVAIS = read_field(keyword='BRAVAIS')

                    # read position value(s)
                    # Typically, inputcard has line '<RBASIS>', followed by x linex of 1x3 bravais matrix values.
                    POSITIONS = read_field(keyword='<RBASIS>')

                except FileNotFoundError as err:
                    print(f"{msg_prefix}: {FileNotFoundError.__name__}: Could not retrieve inputcard from its "
                          f"{_VoronoiCalculation.__name__} {vorocalc}.")
                    return

        #######################
        # 3) Recalculate ANG_BOHR_KKR from inputcard alat and bravais
        def reverse_calc_ANG_BOHR_KKR(inp_arr: _np.ndarray, struc_arr: _np.ndarray):
            def reverse_calc_single_ANG_BOHR_KKR(x: float, y: float):
                # print(f'calc ALATBASIS * {x} / {y}')
                return ALATBASIS * x / y if (y != 0.0 and x != 0.0) else 0.0

            if inp_arr.shape == struc_arr.shape:
                ANG_BOHR_KKR_list = [reverse_calc_single_ANG_BOHR_KKR(x, y)
                                     for x, y in _np.nditer([inp_arr, struc_arr])]
                return ANG_BOHR_KKR_list
            else:
                print(f"{msg_prefix}: Shapes of inputcard matrix and structure matrix "
                      f"do not match: {inp_arr.shape} != {struc_arr.shape}.")
                return

        a2b_list = []
        a2b_list.extend(reverse_calc_ANG_BOHR_KKR(BRAVAIS, structure_cell))
        a2b_list.extend(reverse_calc_ANG_BOHR_KKR(POSITIONS, structure_positions))
        a2b_list = _np.array(a2b_list)

        a2b_list = _math_util.drop_values(a2b_list, 'zero', 'nan')

        # print('a2b_list')
        # print(a2b_list)

        ANG_BOHR_KKR = _np.mean(a2b_list)

        #######################
        # 4) Determine constant type from reverse-calculated constant

        difference = _OrderedDict()
        # difference = {}
        for ctype, value in self.ANG_BOHR_KKR.items():
            difference[ctype] = abs(ANG_BOHR_KKR - value)

        # find indices of minima
        indices = [i for i, val in enumerate(difference.values()) if val == min(difference.values())]
        # in case there are more than one minimum, assign by constants type importance order:
        #  lower index = higher importance. But issue a warning.
        constants_version = list(difference.keys())[indices[0]]
        if len(indices) > 1:
            print(f"Info: Workchain {wc} reverse-calculated 'ANG_BOHR_KKR' value undecisive. Could be either of "
                  f"{[list(difference.keys())[i] for i in indices]}. Chose {constants_version}.")

        #######################
        # 5) Add results to dataframe
        if group_label:
            row['group'] = group_label
        row['ctime'] = wc.ctime
        row['ANG_BOHR_KKR'] = ANG_BOHR_KKR
        row['constants_version'] = constants_version
        row['diff_new'] = difference[KkrConstantsVersion.NEW]
        row['diff_old'] = difference[KkrConstantsVersion.OLD]
        row['diff_interim'] = difference[KkrConstantsVersion.INTERIM]

        self._df = self._df.append(_pd.Series(name=wc.uuid, data=row))

        if set_extra:
            extra = {
                'constants_version': constants_version.name,
                'ANG_BOHR_KKR': None,
                'RY_TO_EV_KKR': None
            }

            if constants_version in [KkrConstantsVersion.NEW,
                                     KkrConstantsVersion.INTERIM,
                                     KkrConstantsVersion.OLD]:
                extra['ANG_BOHR_KKR'] = self.ANG_BOHR_KKR[constants_version]
                extra['RY_TO_EV_KKR'] = self.RY_TO_EV_KKR[constants_version]
            else:
                extra['ANG_BOHR_KKR'] = ANG_BOHR_KKR
                extra['RY_TO_EV_KKR'] = None  # TODO recalculate as well

            wc.set_extra('kkr_constants_version', extra)

    def check_workchain_group(self, group: _Group,
                              process_labels: list = [],
                              set_extra: bool = False,
                              zero_threshold: float = 1e-15):
        """Classify a group of finished workchains by their used KKR constants versions by reverse-calculation.

        The result is available as a dataframe :py:attr:`~aiida_jutools.util_kkr.KkrConstantsVersionChecker.df`.

        Note: Will always check constants version by recalculating it, even if it may have already been set as an
        extra. To classify with the extra instead, use method TODO.

        :param group: a group with aiida-kkr workchain nodes. Workchains must have a ``kkr_startpot_wc`` descendant.
        :param process_labels: list of valid aiida-kkr workchain process labels, e.g. ['kkr_scf_wc', ...].
        :param set_extra: True: Set an extra on the workchain denoting the identified KKR constants version and values.
        :param zero_threshold: Set structure cell elements below this threshold to zero to counter rounding errors.
        """
        if not process_labels:
            print("Warning: No process labels specified. I will do nothing. Specify labels of processes which have "
                  "a 'kkr_startpot_wc' descendant. Valid example: ['kkr_scf_wc', 'kkr_imp_wc'].")
        else:
            for node in group.nodes:
                if isinstance(node, _WorkChainNode) and node.process_label in process_labels:
                    self.check_single_workchain(wc=node,
                                                set_extra=set_extra,
                                                zero_threshold=zero_threshold,
                                                group_label=group.label)

    def check_single_workchain_provenance(self, wc: _WorkChainNode):
        """Check whether the workchain and all its ancestors of a workchain used the same KKR constants versions.

        This requires that the constants version on the workchain AND its ancestors was set as extra before with either
        :py:class:`~aiida_jutools.util_kkr.KkrConstantsVersionChecker.check_single_workchain` or
        :py:class:`~aiida_jutools.util_kkr.KkrConstantsVersionChecker.check_workchain_group`.

        Currently, only ``kkr_imp_wc`` workchains supported.

        Currently checked nodes provenance path (these must have the extras): ``kkr_scf_wc`` > ``kkr_imp_wc``.

        In theory, all constants versions along the provenance path MUST be identical, and if not, the workchain
        should have failed. If it has finished successfully however, the extras must be wrong.

        :param wc: finished aiida-kkr workchain. Must have a ``kkr_startpot_wc`` descendant.
        """
        # TODO: Include intermediate kkr_imp_wc in provenance path check (e.g. GF writeout kkr_imp_wc).
        # TODO: store findings in dataframe or dict

        if not wc.process_label == 'kkr_imp_wc':
            print(f"Workchain '{wc.label}', pk={wc.pk} is not a {_kkr_imp_wc.__name__}. Currently not supported.")
        else:
            try:
                imp_version = wc.extras['kkr_constants_version']['constants_version']

                scf_wcs = wc.get_incoming(node_class=_RemoteData,
                                          link_label_filter='remote_data_host').all_nodes()[0].get_incoming(
                    node_class=_kkr_scf_wc).all_nodes()
                if not scf_wcs:
                    print(f"Workchain '{wc.label}', pk={wc.pk} does not have a {_kkr_scf_wc.__name__} ancestor.")
                else:
                    try:
                        scf_version = scf_wcs[0].extras['kkr_constants_version']['constants_version']
                        if imp_version != scf_version:
                            print(f"Mismatch in {KkrConstantsVersion.__name__} extras for kkr_imp_wc pk={wc.pk}, "
                                  f"label='{wc.label}': parent kkr_scf_wc {scf_version}, kkr_imp_wc {imp_version}.")
                    except KeyError as err:
                        print(f"Workchain '{wc.label}', pk={wc.pk} is missing 'kkr_constants_version' extra.")
            except KeyError as err:
                print(f"Workchain '{wc.label}', pk={wc.pk} is missing 'kkr_constants_version' extra.")

    def check_workchain_group_provenance(self, group: _Group,
                                         process_labels: list = ['kkr_imp_wc']):
        """Check whether the workchain and all its ancestors of a workchain used the same KKR constants versions.

        This requires that the constants version on the workchain AND its ancestors was set as extra before with either
        :py:class:`~aiida_jutools.util_kkr.KkrConstantsVersionChecker.check_single_workchain` or
        :py:class:`~aiida_jutools.util_kkr.KkrConstantsVersionChecker.check_workchain_group`.

        Currently, only ``kkr_imp_wc`` workchains supported.

        Currently checked nodes provenance path (these must have the extras): ``kkr_scf_wc`` > ``kkr_imp_wc``.

        Currently, findings are only printed, not stored in any way.

        In theory, all constants versions along the provenance path MUST be identical, and if not, the workchain
        should have failed. If it has finished successfully however, the extras must be wrong.

        :param group: a group with aiida-kkr workchain nodes. Workchains must have a ``kkr_startpot_wc`` descendant.
        :param process_labels: currently only ['kkr_imp_wc'] supported.
        """
        if not process_labels or process_labels != ['kkr_imp_wc']:
            print("Warning: Unsupported process_labels list. I will do nothing. Currently supported: ['kkr_imp_wc'].")
        else:
            for node in group.nodes:
                if isinstance(node, _WorkChainNode) and node.process_label in process_labels:
                    self.check_single_workchain_provenance(node)

