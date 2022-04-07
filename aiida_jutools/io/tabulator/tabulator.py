# -*- coding: utf-8 -*-
# pylint: disable=unused-import
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
"""Tools for working with AiiDA IO: tabulation: Tabulator."""
from __future__ import annotations


import copy
import json
from typing import Any, Iterable

from aiida import orm
import pandas as pd
from masci_tools.util import python_util as _masci_python_util
from masci_tools.io.parsers.tabulator import Recipe, Tabulator, TableType

import aiida_jutools as _jutools


class NodeTabulator(Tabulator):
    """For tabulation of a collection of nodes' (of same type) properties into a dict or dataframe.

    Class extends :py:class:`~.Tabulator`. See also its docstring.

    The top-level properties tabulated by default can be seen by calling :py:attr:`~node_type_include_list`.

    From `.inputs` / `.outputs` nodes, only properties (attributes) of :py:class:`~aiida.orm.Dict` nodes get
    included in the tabulation.

    Nested properties get unpacked and their subproperties tabulated unto a certain level.

    When defining include list or exclude list, this unpacking limit has to be taken into accound.

    In the transformer, a higher-level nested property may shadow subproperties such that they do not get
    unpacked or transformed.

    TODO: increase memory performance:

    - use optional dtypes from recipe (see TODO in Recipe) when building table. otherwise, e.g. when returning
      pandas dataframe, all columns will have dtype 'object' or 'float64' and the table won't fit into memory
      anymore very quickly.
    - internal storage format dict of lists while building must remain, but when finished, convert to dict
      of numpy arrays -> more memory efficient. for repeated tabulate() calls (to build larger table), have
      to adjust code to concatenate new lists to existing numpy arrays when finished.
    - change tabulate() signature: should not return table, only build it. another method (e.g. table @property
      getter) should return table and before del (delete) its inner storage (self._table) of it, because return
      will likely create a copy. that way can ~half the memory footprint.
    - when returning pandas dataframe, and recipe supplied no dtypes, use automatic downcasting to smallest dtype
      for numeric types (pd.to_numeric), and maybe 'categorical' for string coluns (if num unique values << num
      values). See pandas > Scaling to larg datasets for more.
    - maybe add save option (or method) and read() method to read in tabulated table. for pandas, that allow a user
      to easily reuse the dtypes information from the recipe.
    """

    def __init__(self,
                 recipe: Recipe = None):
        """Init node tabulator.

        Class extends :py:class:`~.Tabulator`. See also its docstring.

        The attributes :py:attr:`~.include_list` and :py:attr:`~.exclude_list` control whic properties
        are to be tabulated. They may be set in a derived class definition, or at runtime on an instance.

        Subclasses define the nature of the objects to be tabulated by making assumptions on their
        property structure. That way, if both include and exclude list are empty, by default the 'complete'
        set of properties of the objects will be tabulated, where the subclass defines the notion of 'complete'.

        If neither exlude nor include list is given, the full set of properties according to implementation
        will be tabulated.

        :param recipe: A recipe with optionally things like include list, exclude list, transformer.
        :param unpack_dicts_level: Include dict properties up to this nesting level.
        :param unpack_inputs_level: Include inputs properties up to this nesting level.
        :param unpack_outputs_level: Include outputs properties up to this nesting level.
        :param kwargs: Additional keyword arguments for subclasses.
        """
        super().__init__(recipe=recipe)

        self._autolist_search_paths = {
            orm.Node: [
                'uuid',
                'label',
                'extras',
            ],
            orm.ProcessNode: [
                'inputs',
                'outputs',
                'process_label',
                'process_state',
                'exit_status'
            ],
            orm.StructureData: [
                "attributes",
                "cell",
                "cell_angles",
                "cell_lengths",
                "get_cell_volume",
                "get_formula",
                "kinds",
                "pbc",
                "sites",
            ]
        }
        self._autolist_unpack_levels = {
            dict: 2,
            orm.Dict: 2,
            'inputs': 3,
            'outputs': 3
        }

    @property
    def autolist_search_paths(self) -> dict[type[orm.Node], list[str]]:
        """The autolist search paths is a list of node types and top-level property string names
        (node attributes).
        The autolist method only searches down these top-level paths."""
        return self._autolist_search_paths

    @autolist_search_paths.setter
    def autolist_search_paths(self,
                              search_paths: dict[type[orm.Node], list[str]]):
        assert all(issubclass(key, orm.Entity) for key in search_paths.keys())
        assert all(isinstance(value, list) for value in search_paths.values())

        self._autolist_search_paths = search_paths

    @property
    def autolist_unpack_levels(self) -> dict[Any, int]:
        """The autolist unpack levels specify at which nesting level the autolist method should stop to
        unpack properties."""
        return self._autolist_unpack_levels

    @autolist_unpack_levels.setter
    def autolist_unpack_levels(self,
                               unpack_levels: dict[Any, int]):
        assert all(isinstance(value, int) for value in unpack_levels.values())
        assert all(key in unpack_levels for key in [dict, orm.Dict, 'inputs', 'outputs'])

        self._autolist_unpack_levels = unpack_levels

    def autolist(self,
                 item: orm.Node,
                 overwrite: bool = False,
                 pretty_print: bool = False,
                 **kwargs: Any):
        """Auto-generate an include list of properties to be tabulated from a given object.

        This can serve as an overview for customized include and exclude lists.

        :param obj: An example object of a type compatible with the tabulator.
        :param overwrite: True: replace recipe list with the auto-generated list. False: Only if recipe list empty.
        :param pretty_print: True: Print the generated list in pretty format.
        :param kwargs: Additional keyword arguments for subclasses.
        """
        if not isinstance(item, orm.Node):
            return
        # get all Dict input/output node names.
        node = item

        include_list = {}

        for node_type, attr_names in self._autolist_search_paths.items():
            if isinstance(node, node_type):
                for attr_name in attr_names:
                    try:
                        attr = getattr(node, attr_name)
                    except AttributeError as err:
                        print(f"Warning: Could not get attr '{attr_name}'. Skipping.")
                        continue

                    is_inputs = attr_name == 'inputs'
                    is_outputs = attr_name == 'outputs'
                    is_dict = isinstance(attr, (dict, orm.Dict)) and attr_name not in ['inputs', 'outputs']
                    is_special = (is_dict or is_inputs or is_outputs)

                    if not is_special:
                        include_list[attr_name] = None
                        continue

                    # now handle the special cases

                    if is_dict:
                        # for instance: node.extras.
                        # note: in future, could use ExtraForm sets here for standardized extras.
                        # get dict structure up to the specified unpacking leve
                        is_aiida_dict = isinstance(attr, orm.Dict)
                        attr = attr.attributes if is_aiida_dict else attr

                        to_level = self.autolist_unpack_levels[type(attr)]
                        props = _masci_python_util.modify_dict(a_dict=attr,
                                                               transform_value=lambda v: None,
                                                               to_level=to_level)
                        if is_aiida_dict:
                            include_list[attr_name] = {'attributes': copy.deepcopy(props)}
                        else:
                            include_list[attr_name] = copy.deepcopy(props)

                    if is_inputs or is_outputs:
                        # get all Dict output link triples
                        link_triples = node.get_incoming(node_class=orm.Dict).all() \
                            if is_inputs else node.get_outgoing(node_class=orm.Dict).all()

                        # Now get all keys in all input/output `Dicts`, sorted alphabetically.
                        all_io_dicts = {lt.link_label: lt.node.attributes for lt in link_triples}

                        # now get structure for all the inputs/outputs
                        in_or_out = 'inputs' if is_inputs else 'outputs'
                        to_level = self.autolist_unpack_levels[in_or_out]
                        props = {
                            link_label: _masci_python_util.modify_dict(a_dict=out_dict,
                                                                       transform_value=lambda v: None,
                                                                       to_level=to_level)
                            for link_label, out_dict in all_io_dicts.items()
                        }
                        include_list[attr_name] = copy.deepcopy(props)

        if pretty_print:
            print(json.dumps(include_list,
                             indent=4))

        if overwrite or not self.recipe.include_list:
            self.recipe.include_list = include_list

    def tabulate(self,
                 collection: Iterable[orm.Node] | orm.Group,
                 table_type: TableType = pd.DataFrame,
                 append: bool = True,
                 column_policy: str = 'flat',
                 pass_item_to_transformer: bool = True,
                 drop_empty_columns: bool = True,
                 verbose: bool = True,
                 **kwargs) -> TableType:
        """This method extends :py:meth:`~.Tabulator.tabulate`. See also its docstring.

        For unpacking standardized extras, .:py:class:`~aiida_jutools.meta.extra.ExtraForm` sets may be used.

        :param collection: group or list of nodes.
        :param table_type: table as pandas DataFrame or dict.
        :param append: True: append to table if not empty. False: Overwrite table.
        :param column_policy: 'flat': Flat table, column names are last keys per keypath, name conflicts produce 
                              warnings. 'flat_full_path': Flat table, column names are full keypaths, 
                              'multiindex': table with MultiIndex columns (if pandas: `MultiIndex` columns), reflecting 
                              the full properties' keypath hierarchies.
        :param pass_node_to_transformer: True: Pass current node to transformer. Enables more complex transformations,
                                         but may be slower for large collections.
        :param drop_empty_columns: Drop None/NaN-only columns. These could
        :param verbose: True: Print warnings.
        :param kwargs: Additional keyword arguments for subclasses.
        :return: Tabulated objects' properties as dict or pandas DataFrame.
        """

        if isinstance(collection, orm.Group):
            collection = collection.nodes

        return super().tabulate(collection,
                                table_type=table_type,
                                append=append,
                                column_policy=column_policy,
                                pass_item_to_transformer=pass_item_to_transformer,
                                drop_empty_columns=drop_empty_columns,
                                verbose=verbose,
                                **kwargs)


    def get_value(item, keypath):
        value, err = _jutools.node.get_from_nested_node(node=item,
                                                        keypath=keypath)
        
        if err:
            return None
        return value