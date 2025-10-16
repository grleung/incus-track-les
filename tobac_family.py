# sean v5

import pandas as pd
import xarray as xr
from typing import Literal
import numpy as np
import copy
import skimage


def identify_feature_families(
    feature_df: pd.DataFrame,
    in_segmentation: xr.DataArray,
    return_grid: bool = False,
    family_column_name: str = "feature_family_id",
    unsegmented_point_values: int = 0,
    below_threshold_values: int = -1,
):
    """
    Function to identify families/storm systems by identifying where segmentation touches.
    At a given time, segmentation areas are considered part of the same family if they
    touch at any point.

    Parameters
    ----------
    feature_df: pd.DataFrame
        Input feature dataframe
    in_segmentation: xr.DataArray
        Input segmentation
    return_grid: bool
        Whether to return the segmentation grid showing families
    family_column_name: str
        The name in the output dataframe of the family ID
    unsegmented_point_values: int
        The value in the input segmentation for unsegmented but above threshold points
    below_threshold_values: int
        The value in the input segmentation for below threshold points

    Returns
    -------
    pd.DataFrame and xr.DataArray or pd.DataFrame
        Input dataframe with family IDs associated with each feature
        if return_grid is True, the segmentation grid showing families is
        also returned.

    """

    # TODO: This does not currently work if you have segmentation data that
    # uses feature numbers pre-merging

    # we need to label the data, but we currently label using skimage label, not dask label.

    # 3D should be 4-D (time, then 3 spatial).
    # 2D should be 3-D (time, then 2 spatial)
    is_3D = len(in_segmentation.shape) == 4
    seg_family_dict = dict()
    out_families = copy.deepcopy(in_segmentation)
    max_family_number = 0
    enable_family_statistics = False  # True

    if enable_family_statistics:
        region_props_vals = ["bbox", "centroid", "num_pixels"]
        family_stats = dict()

    for time_index in range(in_segmentation.shape[0]):
        in_arr = np.array(in_segmentation.values[time_index])

        segmented_arr = np.logical_and(
            in_arr != unsegmented_point_values, in_arr != below_threshold_values
        )
        # These are our families
        family_labeled_data, number_families = skimage.measure.label(
            segmented_arr, return_num=True
        )
        if enable_family_statistics:
            all_family_nums = list()

            family_props = skimage.measure.regionprops(family_labeled_data)
            for family in family_props:
                all_family_nums.append(family.label + max_family_number)
                family_stats[family.label + max_family_number] = dict()
                # family_stats[family.label+max_family_number]['frame'] =

                family_stats[family.label + max_family_number]["num_pixels"] = (
                    family["num_pixels"]
                )
                family_stats[family.label + max_family_number][
                    family_column_name
                ] = (family.label + max_family_number)
                if not is_3D:
                    family_stats[family.label + max_family_number][
                        "hdim_1_center"
                    ] = family["centroid"][0]
                    family_stats[family.label + max_family_number][
                        "hdim_2_center"
                    ] = family["centroid"][1]

                else:
                    # TODO: integrate 3D stats - mostly around center coordinates - need the functions in tobac proper
                    raise NotImplementedError("3D stats not implemented yet")

        # now we need to note feature->family relationship in the dataframe.
        segmentation_props = skimage.measure.regionprops(in_arr)

        # associate feature ID -> family ID
        for seg_area in segmentation_props:
            if is_3D:
                seg_family = family_labeled_data[
                    seg_area.coords[0, 0],
                    seg_area.coords[0, 1],
                    seg_area.coords[0, 2],
                ]
            else:
                seg_family = family_labeled_data[
                    seg_area.coords[0, 0], seg_area.coords[0, 1]
                ]
            seg_family_dict[seg_area.label] = seg_family + max_family_number
        """
        if seg_area is not None and enable_family_statistics:
            curr_feat = seg_area.label
            curr_frame = feature_df[feature_df['feature'] == curr_feat]['frame']
            for family_id in all_family_nums:
                family_stats[family_id]['frame'] = curr_frame.values[0]

        """
        out_families[time_index] = family_labeled_data + max_family_number * (
            (family_labeled_data > unsegmented_point_values).astype(int)
        )
        max_family_number = max_family_number + number_families

    family_series = pd.Series(seg_family_dict, name=family_column_name)
    feature_series = pd.Series(
        {x: x for x in seg_family_dict.keys()}, name="feature"
    )
    family_df = pd.concat([family_series, feature_series], axis=1)
    out_df = feature_df.merge(family_df, on="feature", how="inner")

    if enable_family_statistics:
        family_stats_df = pd.DataFrame.from_dict(family_stats, orient="index")

    if return_grid:
        if enable_family_statistics:
            return out_df, family_stats_df, out_families

    else:
        if enable_family_statistics:
            return out_df, family_stats_df
        else:
            return out_df


def track_feature_families(
    in_feat_arr: pd.DataFrame,
    in_family_stat_arr: pd.DataFrame,
    maintain_family_metric: Literal["cells", "area",'cmf_volume'] = "area",
    feat_family_column_name: str = "feature_family_id",
    cell_column_name: str = "cell",
    tracked_family_column_name: str = "tracked_family_id",
):

    time_col_name: str = "time"

    family_ids_of_interest = []  # [139140, 139055]

    # number to start the new tracking ID with
    curr_tracked_family_num = 1

    # the fraction of previous family cells to retain to be considered part of the same track
    maintain_track_cell_fraction = 0.5

    # the fraction of previous family area to retain to be considered part of the same track
    maintain_track_area_fraction = 0.3

    if maintain_track_cell_fraction < 0.5:
        raise ValueError(
            "Cannot guarantee unique track with maintain_track_cell_fraction  <0.5 "
        )

    # relates an individual feature family at a given time to a tracked feature family
    family_to_track_relationship = dict()

    # list of mergers/splits
    merge_split_list = list()

    # get all frame numbers we are dealing with
    all_feat_frames = sorted(in_feat_arr["frame"].unique())

    # we don't need all variables here, just some of them. cut down to save computation time
    reduced_feat_arr = in_feat_arr[
        ["frame", "idx", "feature", cell_column_name, feat_family_column_name]
    ]
    for i, frame_num in enumerate(all_feat_frames):
        # pull the current time
        curr_time_feat_arr = reduced_feat_arr[
            reduced_feat_arr["frame"] == frame_num
        ]
        # get the relationship between feature family at this time -> cell at this time
        sets_curr_time = curr_time_feat_arr.groupby(feat_family_column_name)[
            "cell"
        ].agg(set)

        # first time we are dealing with
        if i == 0:
            # just set all feature families to the same tracked number
            sets_prev_time = copy.deepcopy(sets_curr_time)
            family_to_track_relationship = {
                sets_curr_time.index[tracked_num]: tracked_num
                + curr_tracked_family_num
                for tracked_num, _ in enumerate(sets_curr_time)
            }
            curr_tracked_family_num += len(sets_curr_time)
            continue

        # from the perspective of the previous time, what is the intersection between cells at t=0 and t=1?
        family_intersections_through_time_prev = dict()
        # from the perspective of the current time, what is the intersection between cells at t=0 and t=1?
        family_intersections_through_time_curr = dict()
        # from the perspective of the
        # loop through all families in the previous timestep
        for i_prev, prev_family_cells in enumerate(sets_prev_time):
            prev_family_label = sets_prev_time.index[i_prev]

            # loop through all families in the current timestep
            for i_curr, curr_family_cells in enumerate(sets_curr_time):
                curr_family_label = sets_curr_time.index[i_curr]

                # do the intersection of all cell numbers for a family at the previous timestep
                # with a family at the current timestep
                prev_curr_family_cell_overlap = prev_family_cells.intersection(
                    curr_family_cells
                )
                # if we have any overlap, record it.
                if len(prev_curr_family_cell_overlap) > 0:
                    # record previous-> current links
                    if (
                        prev_family_label
                        not in family_intersections_through_time_prev
                    ):
                        family_intersections_through_time_prev[
                            prev_family_label
                        ] = [
                            (curr_family_label, prev_curr_family_cell_overlap),
                        ]
                    else:
                        family_intersections_through_time_prev[
                            prev_family_label
                        ].append(
                            (curr_family_label, prev_curr_family_cell_overlap)
                        )

                    # record current ->previous links
                    if (
                        curr_family_label
                        not in family_intersections_through_time_curr
                    ):
                        family_intersections_through_time_curr[
                            curr_family_label
                        ] = [
                            (prev_family_label, prev_curr_family_cell_overlap),
                        ]
                    else:
                        family_intersections_through_time_curr[
                            curr_family_label
                        ].append(
                            (prev_family_label, prev_curr_family_cell_overlap)
                        )

        # now that we have identified intersections, let's process them.
        for i_curr, curr_family_cells in enumerate(sets_curr_time):
            curr_family_label = sets_curr_time.index[i_curr]
            if (
                curr_family_label not in family_intersections_through_time_curr
                or len(
                    family_intersections_through_time_curr[curr_family_label]
                )
                == 0
            ):
                # we have no common cells between the current time and the previous time
                # give this family a new track number
                family_to_track_relationship[curr_family_label] = (
                    curr_tracked_family_num
                )

                # increment current family track number
                curr_tracked_family_num += 1
                if curr_family_label in family_ids_of_interest:
                    print("Crashing out at no common cells")

            elif (
                len(family_intersections_through_time_curr[curr_family_label])
                == 1
            ):
                # we have only one family that shares cells from the previous time with the current time
                prev_family_label = family_intersections_through_time_curr[
                    curr_family_label
                ][0][0]
                # check if the previous family can only be linked to this cell.
                # print(prev_family_label, family_intersections_through_time_curr[curr_family_label])
                if (
                    len(
                        family_intersections_through_time_prev[
                            prev_family_label
                        ]
                    )
                    == 1
                ):
                    family_to_track_relationship[curr_family_label] = (
                        family_to_track_relationship[prev_family_label]
                    )
                else:
                    # this family can only be linked to one previous cell, but the previous cell could be
                    # linked to multiple current families (split)
                    common_cells_prev_curr = (
                        family_intersections_through_time_curr[
                            curr_family_label
                        ][0][1]
                    )
                    all_prev_cells = sets_prev_time[prev_family_label]

                    # check if we can maintain family number based on our metric
                    keep_track_number = False
                    if maintain_family_metric == "cells":
                        # check if our fraction of cells in the current time family is > our prescribed fraction
                        keep_track_number = (
                            len(common_cells_prev_curr) / len(all_prev_cells)
                            > maintain_track_cell_fraction
                        )
                    elif maintain_family_metric == "area":
                        # check if our fraction of area maintained in the current family is > our prescribed fraction
                        curr_family_area = in_family_stat_arr.at[
                            curr_family_label, "num_pixels"
                        ]
                        prev_family_area = in_family_stat_arr.at[
                            prev_family_label, "num_pixels"
                        ]
                        keep_track_number = (
                            prev_family_area / curr_family_area
                        ) > maintain_track_area_fraction and (
                            curr_family_area / prev_family_area
                        ) > maintain_track_area_fraction
                        if keep_track_number:
                            pass
                            # print("Keeping track number. Location 1. prev_family: {0}, curr_family: {1}, prev_area: {2}, curr_area: {3}, prev_cells: ".format(
                            #    prev_family_label, curr_family_label, prev_family_area, curr_family_area
                            # ))

                    elif maintain_family_metric=='cmf_volume':
                        # check if our fraction of area maintained in the current family is > our prescribed fraction
                        curr_family_area = in_family_stat_arr.at[
                            curr_family_label, "cmf_volume"
                        ]
                        prev_family_area = in_family_stat_arr.at[
                            prev_family_label, "cmf_volume"
                        ]
                        keep_track_number = (
                            prev_family_area / curr_family_area
                        ) > maintain_track_area_fraction and (
                            curr_family_area / prev_family_area
                        ) > maintain_track_area_fraction
                        if keep_track_number:
                            pass

                    else:
                        raise ValueError(
                            "Only acceptable metrics are 'cells' and 'area'."
                        )

                    if keep_track_number:
                        # if it is, we can maintain the track number.
                        family_to_track_relationship[curr_family_label] = (
                            family_to_track_relationship[prev_family_label]
                        )
                    else:
                        if curr_family_label in family_ids_of_interest:
                            print("Crashing out at too few cells")

                        # if it has too few cells from the previous family, it gets a new/unique number
                        family_to_track_relationship[curr_family_label] = (
                            curr_tracked_family_num
                        )
                        curr_tracked_family_num += 1
                        # log the split.
                        merge_split_list.append(
                            (
                                family_to_track_relationship[prev_family_label],
                                family_to_track_relationship[curr_family_label],
                            )
                        )

            else:
                # >1 family that shares cells from the previous time with the current time
                prev_family_label = family_intersections_through_time_curr[
                    curr_family_label
                ][0][0]

                # check if this family's parents all only have one child
                children_family_ancestors = {
                    cell_intersect_pair[0]: cell_intersect_pair[1]
                    for cell_intersect_pair in family_intersections_through_time_curr[
                        curr_family_label
                    ]
                }
                max_parent_children = max(
                    [
                        len(children_family_ancestors[x])
                        for x in children_family_ancestors
                    ]
                )
                all_curr_cells = sets_curr_time[curr_family_label]
                if max_parent_children == 1:
                    # all parents only have this family as child cell.
                    # many (prev) : one (curr) relationship

                    found_parent_link = False
                    largest_prev_family_area = -1.0
                    largest_prev_family_id = -1
                    for (
                        parent_cell_label,
                        parent_cell_intersection,
                    ) in family_intersections_through_time_curr[
                        curr_family_label
                    ]:

                        keep_track_number = False
                        if maintain_family_metric == "cells":
                            # Link only if one parent has > maintain_track_cell_fraction of the child's cells
                            # this can only be triggered once
                            keep_track_number = (
                                len(parent_cell_intersection)
                                / len(all_curr_cells)
                            ) > maintain_track_cell_fraction
                            family_to_track_relationship[curr_family_label] = (
                                family_to_track_relationship[parent_cell_label]
                            )
                            found_parent_link = True
                            break

                        elif maintain_family_metric == "area":
                            # check if our fraction of area maintained in the current family is > our prescribed fraction
                            curr_family_area = in_family_stat_arr.at[
                                curr_family_label, "num_pixels"
                            ]
                            prev_family_area = in_family_stat_arr.at[
                                parent_cell_label, "num_pixels"
                            ]
                            keep_track_number = (
                                prev_family_area / curr_family_area
                            ) > maintain_track_area_fraction and (
                                curr_family_area / prev_family_area
                            ) > maintain_track_area_fraction

                            if keep_track_number:
                                if prev_family_area > largest_prev_family_area:
                                    largest_prev_family_area = prev_family_area
                                    largest_prev_family_id = (
                                        family_to_track_relationship[
                                            parent_cell_label
                                        ]
                                    )

                        else:
                            raise ValueError(
                                "Only acceptable metrics are 'cells' and 'area'."
                            )

                    # check that we are keeping largest family
                    if (
                        maintain_family_metric == "area"
                        and largest_prev_family_id != -1
                    ):
                        family_to_track_relationship[curr_family_label] = (
                            largest_prev_family_id
                        )
                        found_parent_link = True

                    if not found_parent_link:
                        # all parents share less than maintain_track_cell of their child's cells.
                        # issue this a new cell number.
                        if curr_family_label in family_ids_of_interest:
                            print("Crashing out at too small area")

                        family_to_track_relationship[curr_family_label] = (
                            curr_tracked_family_num
                        )
                        curr_tracked_family_num += 1
                    # record the merger, regardless.
                    for (
                        parent_cell_label,
                        _,
                    ) in family_intersections_through_time_curr[
                        curr_family_label
                    ]:
                        merge_split_list.append(
                            (
                                family_to_track_relationship[parent_cell_label],
                                family_to_track_relationship[curr_family_label],
                            )
                        )

                else:
                    # at least one parent has multiple children, meaning we have a many:many relationship.
                    # We will only link if there is a parent cell where > maintain_track_cell_fraction of the cells in the child are there
                    # AND if the opposite fraction is also true
                    all_prev_cells = sets_prev_time[prev_family_label]
                    found_parent_link = False
                    largest_prev_family_area = -1.0
                    largest_prev_family_id = -1

                    for (
                        parent_cell_label,
                        parent_cell_intersection,
                    ) in family_intersections_through_time_curr[
                        curr_family_label
                    ]:

                        keep_track_number = False
                        if maintain_family_metric == "cells":
                            # Link only if one parent has > maintain_track_cell_fraction of the child's cells
                            keep_track_number = (
                                len(parent_cell_intersection)
                                / len(all_curr_cells)
                            ) > maintain_track_cell_fraction and (
                                len(parent_cell_intersection)
                                / len(all_prev_cells)
                            ) > maintain_track_cell_fraction
                            family_to_track_relationship[curr_family_label] = (
                                family_to_track_relationship[parent_cell_label]
                            )
                            found_parent_link = True

                        elif maintain_family_metric == "area":
                            # check if our fraction of area maintained in the current family is > our prescribed fraction

                            curr_family_area = in_family_stat_arr.at[
                                curr_family_label, "num_pixels"
                            ]
                            prev_family_area = in_family_stat_arr.at[
                                parent_cell_label, "num_pixels"
                            ]
                            keep_track_number = (
                                prev_family_area / curr_family_area
                            ) > maintain_track_area_fraction and (
                                curr_family_area / prev_family_area
                            ) > maintain_track_area_fraction

                            if curr_family_label in family_ids_of_interest:
                                print(
                                    f"parent_cell_label: {parent_cell_label}, curr_family: {curr_family_label}, "
                                    f"prev_family_area: {prev_family_area}, curr_family_area: {curr_family_area},"
                                )

                            if keep_track_number:
                                if prev_family_area > largest_prev_family_area:
                                    largest_prev_family_area = prev_family_area
                                    largest_prev_family_id = (
                                        family_to_track_relationship[
                                            parent_cell_label
                                        ]
                                    )

                        else:
                            raise ValueError(
                                "Only acceptable metrics are 'cells' and 'area'."
                            )

                    # check that we are keeping largest family
                    if (
                        maintain_family_metric == "area"
                        and largest_prev_family_id != -1
                    ):
                        family_to_track_relationship[curr_family_label] = (
                            largest_prev_family_id
                        )
                        found_parent_link = True
                    if not found_parent_link:
                        # all parents share less than maintain_track_cell of their child's cells.
                        # issue this a new cell number.
                        if curr_family_label in family_ids_of_interest:
                            print(
                                "Crashing out at too few small area final", ""
                            )

                        family_to_track_relationship[curr_family_label] = (
                            curr_tracked_family_num
                        )
                        curr_tracked_family_num += 1

                    # record the merge/split mess
                    for (
                        parent_cell_label,
                        _,
                    ) in family_intersections_through_time_curr[
                        curr_family_label
                    ]:
                        merge_split_list.append(
                            (
                                family_to_track_relationship[parent_cell_label],
                                family_to_track_relationship[curr_family_label],
                            )
                        )

        # end of loop
        sets_prev_time = sets_curr_time

    # we have links between families and tracked families.
    # TODO: make into a frame again.
    out_df = copy.deepcopy(in_feat_arr)
    tracked_family_arr = out_df[feat_family_column_name].apply(
        lambda x: family_to_track_relationship[x]
    )
    out_df[tracked_family_column_name] = tracked_family_arr

    return out_df


def vec_translate(array: np.array, my_dict: dict):
    return np.vectorize(my_dict.__getitem__)(array)


def reassign_grid_coords_to_family(
    in_family_track_df: pd.DataFrame,
    in_grid: xr.DataArray,
    family_id_name: str = "feature_family_id",
    tracked_family_id_name: str = "tracked_family_id",
):
    """
    Function to renumber the input (i.e., per-frame) family ID to a per tracked family ID.

    Parameters
    ----------
    in_family_track_df: pd.DataFrame
        An output tracked dataframe from track_feature_families
    in_grid: xr.DataArray
        An output grid from family tracking
    family_id_name: str
        The name of the original untracked family ID column
    tracked_family_id_name: str
        The name of the tracked family ID column

    Returns
    -------
    return_grid: xr.DataArray
        A grid with new renumbered values to reflect the tracked families
    """
    # feat_tracked_families.groupby(['frame','feature_family_id'])['tracked_family_id'].max().to_dict()

    # return_grid = copy.deepcopy(in_grid)

    dims_reordered = ["time"]
    for curr_dim in in_grid.dims:
        if curr_dim != "time":
            dims_reordered.append(curr_dim)
    return_grid = copy.deepcopy(in_grid.transpose(*dims_reordered))
    for i, (curr_time, grid_at_time) in enumerate(
        return_grid.groupby("time", squeeze=False)
    ):
        curr_frame_seg = in_family_track_df[
            in_family_track_df["time"] == curr_time
        ]

        map_family_to_track_id = {x: 0 for x in np.unique(grid_at_time.values)}
        map_family_to_track_id.update(
            dict(
                curr_frame_seg.groupby(
                    [family_id_name, tracked_family_id_name]
                )["frame"]
                .count()
                .index.values
            )
        )

        return_grid.values[i] = vec_translate(
            return_grid.values[i], map_family_to_track_id
        )

    return_grid = return_grid.rename("tracked_family_id")

    return return_grid


def combine_feature_families(
    in_feature_dfs: list[pd.DataFrame],
    in_stats: list[pd.DataFrame],
    in_grid: list[xr.DataArray] = None,
    renumber_features: bool = False,
    old_feature_column_name=None,
    renumber_families: bool = True,
    old_family_column_name: str = "feature_family_id_original",
    family_column_name: str = "feature_family_id",
):
    """
    Function to combine dataframes of separately calculated feature families into one unified
    dataframe for features.

    Parameters
    ----------
    in_feature_dfs: list[pd.DataFrame]
        List of dataframes generated by `identify_feature_families`
    in_stats: list[pd.DataFrame]
         List of statistics dataframes generated by `identify_feature_families`
    in_grid: list[xr.DataArray], optional
        List of DataArrays of the grids output by `identify_feature_families`. Warning that combining these is
        computationally expensive.
    renumber_features: bool, optional (default: False)
        If true, features are renumber with contiguous integers. If false, the
        old feature numbers will be retained, but an exception will be raised if
        there are any non-unique feature numbers. If you have non-unique feature
        numbers and want to preserve them, use the old_feature_column_name to
        save the old feature numbers to under a different column name.
    old_feature_column_name: str or None, optional (default: None)
        The column name to preserve old feature numbers in. If None, these
        old numbers will be deleted. Users may want to enable this feature
        if they have run segmentation with the separate dataframes and
        therefore old feature numbers.
    renumber_families: bool, optional (default: True)
        If true, families are renumbered with contiguous integers. If false, the
        old family numbers will be retained, but an exception will be raised if
        there are any non-unique family numbers. If you have non-unique family
        numbers and want to preserve them, use the old_family_column_name to
        save the old family numbers to under a different column name.
    old_family_column_name: str or None, optional (default: "feature_family_id_original")
        The column name to preserve old family numbers in. If None, these
        old numbers will be deleted. Users may want to enable this feature
        if they have family identification with grid output enabled with the separate dataframes and
        therefore old family numbers.

    family_column_name: str
        The name in the output dataframe of the family ID



    Returns
    -------
    pd.DataFrame, pd.DataFrame, Optional xr.DataArray
        Combined feature dataframe, combined statistics dataframe, and combined grid if passed in.

    """

    if in_grid is not None:
        raise NotImplementedError("Merging of grids is not yet supported")

    # combine the feature dataframes
    combined_feature_df = tobac.utils.general.combine_feature_dataframes(
        in_feature_dfs,
        renumber_features=renumber_features,
        old_feature_column_name=old_feature_column_name,
    )

    # get time: frame mapping
    time_frame_map = family_stats.groupby("time")["frame"].max().to_dict()

    # now need to combine the stats dataframes
    combined_stats_df = pd.concat(in_stats)

    combined_stats_df["frame"] = [
        time_frame_map[x] for x in combined_stats_df["time"]
    ]

    if not renumber_families and np.any(
        np.bincount(
            combined_feature_df[family_column_name]
            + np.nanmin(combined_feature_df[family_column_name])
        )
        > 1
    ):
        raise ValueError(
            "Non-unique family values detected. Combining feature dataframes with original feature numbers"
            " cannot be performed because duplicate feature numbers exist, please use 'renumber_features=True'. "
            "If you would like to preserve the original feature numbers, please use the 'old_feature_column_name' "
            "keyword to define a new column for these values in the returned dataframe"
        )

    if old_feature_column_name is not None:
        combined_feature_df[old_family_column_name] = copy.deepcopy(
            combined_feature_df[family_column_name]
        )
        combined_stats_df[old_family_column_name] = copy.deepcopy(
            combined_stats_df[family_column_name]
        )

    new_family_numbers = np.empty(
        len(combined_feature_df[family_column_name]),
        dtype=combined_feature_df[family_column_name].dtype,
    )
    new_family_numbers_stats = np.empty(
        len(combined_stats_df[family_column_name]),
        dtype=combined_stats_df[family_column_name].dtype,
    )
    family_number_map = dict()
    # let's start with the first frame's minimum number.
    min_family_num = combined_feature_df[
        combined_feature_df["frame"] == combined_feature_df["frame"].min()
    ][family_column_name].min()
    for i, (index, row) in enumerate(combined_feature_df.iterrows()):
        curr_fam_pair = (row["frame"], row[family_column_name])
        if curr_fam_pair in family_number_map:
            new_family_numbers[i] = family_number_map[curr_fam_pair]
        else:
            new_family_numbers[i] = min_family_num
            family_number_map[curr_fam_pair] = min_family_num
            min_family_num += 1

    for i, (index, row) in enumerate(combined_stats_df.iterrows()):
        curr_fam_pair = (row["frame"], row[family_column_name])
        new_family_numbers_stats[i] = family_number_map[curr_fam_pair]

    combined_feature_df[family_column_name] = new_family_numbers
    combined_stats_df[family_column_name] = new_family_numbers_stats

    combined_stats_df = combined_stats_df.reset_index()
    return combined_feature_df, combined_stats_df
