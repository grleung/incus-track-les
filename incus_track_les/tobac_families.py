import numpy as np
import pandas as pd
import xarray as xr
import copy
from typing import Literal, Optional, Union
import skimage
import datetime
import sys
import cftime

import tobac
import tobac.utils.internal.label_props as label_props
import tobac.utils.periodic_boundaries as pbc_utils
import tobac.utils.datetime as datetime_utils


def renumber_second_df_for_concat(
    df1,
    df2,
    time_col="time",
    frame_col="frame",
    feature_col="feature",
    family_col="feature_family_id",
):
    """ 
    Renumber only df2 so it can be concatenated after df1.

    Assumptions:
    - df1 is already correctly numbered
    - df2 starts its numbering over again and must continue after df1
    - frame in df2 should continue after the last frame in df1
    - feature in df2 is already consecutive (no gaps), so we only shift it
    - feature_family_id in df2 is renumbered by (frame, old_feature_family_id):
        same old family ID in same frame      -> same new family ID
        same old family ID in different frame -> different new family ID
    - old_feature_family_id == -1 is invalid and stays -1
    - NaN family IDs remain NaN
    """

    df1 = df1.copy()
    df2 = df2.copy()

    # ------------------------------------------------------------
    # Preserve original df2 columns exactly once
    # ------------------------------------------------------------
    if frame_col in df2.columns and "old_frame" not in df2.columns:
        df2["old_frame"] = df2[frame_col]

    if feature_col in df2.columns and "old_feature" not in df2.columns:
        df2["old_feature"] = df2[feature_col]

    if family_col in df2.columns and "old_feature_family_id" not in df2.columns:
        df2["old_feature_family_id"] = df2[family_col]

    if "old_frame" not in df2.columns:
        df2["old_frame"] = np.nan

    if "old_feature" not in df2.columns:
        df2["old_feature"] = np.nan

    if "old_feature_family_id" not in df2.columns:
        df2["old_feature_family_id"] = np.nan

    # ------------------------------------------------------------
    # Offsets from df1
    # ------------------------------------------------------------
    if len(df1) == 0:
        frame_offset = 0
        feature_offset = 0
        next_family_id = 1
    else:
        frame_offset = (
            int(df1[frame_col].dropna().max()) + 1
            if frame_col in df1.columns and df1[frame_col].notna().any()
            else 0
        )

        feature_offset = (
            int(df1[feature_col].dropna().max())
            if feature_col in df1.columns and df1[feature_col].notna().any()
            else 0
        )

        valid_family_ids_df1 = (
            df1[family_col].notna() & (df1[family_col] != -1)
            if family_col in df1.columns
            else pd.Series(False, index=df1.index)
        )

        next_family_id = (
            int(df1.loc[valid_family_ids_df1, family_col].max()) + 1
            if valid_family_ids_df1.any()
            else 1
        )

    # ------------------------------------------------------------
    # 1) Shift frame in df2 so it continues after df1
    # ------------------------------------------------------------
    if frame_col in df2.columns:
        df2[frame_col] = df2[frame_col] + frame_offset

    # ------------------------------------------------------------
    # 2) Shift feature in df2 so it continues after df1
    # Since features are already consecutive, no full renumber is needed
    # ------------------------------------------------------------
    if feature_col in df2.columns:
        df2[feature_col] = df2[feature_col] + feature_offset

    # ------------------------------------------------------------
    # 3) Renumber feature_family_id in df2
    # - same (frame, old_feature_family_id) gets same new family ID
    # - -1 stays -1
    # - NaN stays NaN
    # ------------------------------------------------------------
    valid_fam = df2["old_feature_family_id"].notna() & (
        df2["old_feature_family_id"] != -1
    )

    fam_pairs = (
        df2.loc[valid_fam, [frame_col, "old_feature_family_id"]]
        .drop_duplicates()
        .sort_values([frame_col, "old_feature_family_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    fam_pairs[family_col] = np.arange(next_family_id, next_family_id + len(fam_pairs))

    df2 = df2.drop(columns=[family_col], errors="ignore").merge(
        fam_pairs,
        on=[frame_col, "old_feature_family_id"],
        how="left",
        sort=False,
    )

    # Restore invalid family IDs
    invalid_minus1 = df2["old_feature_family_id"] == -1
    df2.loc[invalid_minus1, family_col] = -1

    return df2


def concat_tracking_dfs(
    df1,
    df2,
    time_col="time",
    frame_col="frame",
    feature_col="feature",
    family_col="feature_family_id",
):
    """
    Renumber df2 relative to df1, then concatenate.
    """
    df2_renumbered = renumber_second_df_for_concat(
        df1,
        df2,
        time_col=time_col,
        frame_col=frame_col,
        feature_col=feature_col,
        family_col=family_col,
    )

    df_out = pd.concat([df1, df2_renumbered], ignore_index=True)
    return df_out


def _ensure_column_not_index(df, col_name):
    """
    If col_name is the index, reset it into a normal column.
    """
    df = df.copy()

    if col_name not in df.columns:
        if df.index.name == col_name:
            df = df.reset_index()
        else:
            raise KeyError(f"'{col_name}' is neither a column nor the index.")

    return df


def concat_and_assign_family_ids_from_reference(
    df_a,
    df_b,
    ref_df,
    time_col="time",
    family_col="feature_family_id",
    old_family_col="old_feature_family_id",
):
    """
    Keep df_a family IDs unchanged, assign df_b family IDs from ref_df by
    matching on [time, old_feature_family_id], then concatenate.

    Intended use
    ------------
    - df_a: first family dataframe, already correctly renumbered
    - df_b: second family dataframe to append
    - ref_df: already-renumbered tracking dataframe containing the exact
      mapping from [time, old_feature_family_id] -> feature_family_id

    Rules
    -----
    - feature_family_id may be a column or the index
    - old_feature_family_id is created if missing
    - old_feature_family_id == -1 stays -1
    - valid rows must all match ref_df
    - no NaN values are allowed in output feature_family_id
    """

    # ------------------------------------------------------------
    # Make sure family_col is a normal column
    # ------------------------------------------------------------
    df_a = _ensure_column_not_index(df_a, family_col)
    df_b = _ensure_column_not_index(df_b, family_col)
    ref_df = _ensure_column_not_index(ref_df, family_col)

    df_a = df_a.copy()
    df_b = df_b.copy()
    ref_df = ref_df.copy()

    # ------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------
    for name, df in [("df_a", df_a), ("df_b", df_b), ("ref_df", ref_df)]:
        if time_col not in df.columns:
            raise KeyError(f"'{time_col}' not found in {name}.")

    # old family id may also be index in ref_df
    if old_family_col not in ref_df.columns:
        if ref_df.index.name == old_family_col:
            ref_df = ref_df.reset_index()
        else:
            raise KeyError(
                f"'{old_family_col}' not found in ref_df. "
                "ref_df must contain original family IDs."
            )

    # ------------------------------------------------------------
    # Preserve original family IDs if not already present
    # ------------------------------------------------------------
    if old_family_col not in df_a.columns:
        df_a[old_family_col] = df_a[family_col]

    if old_family_col not in df_b.columns:
        df_b[old_family_col] = df_b[family_col]

    # ------------------------------------------------------------
    # Normalize dtypes
    # ------------------------------------------------------------
    df_a[family_col] = pd.to_numeric(df_a[family_col], errors="raise").astype("Int64")
    df_b[family_col] = pd.to_numeric(df_b[family_col], errors="raise").astype("Int64")
    df_a[old_family_col] = pd.to_numeric(df_a[old_family_col], errors="raise").astype(
        "Int64"
    )
    df_b[old_family_col] = pd.to_numeric(df_b[old_family_col], errors="raise").astype(
        "Int64"
    )
    ref_df[family_col] = pd.to_numeric(ref_df[family_col], errors="raise").astype(
        "Int64"
    )
    ref_df[old_family_col] = pd.to_numeric(
        ref_df[old_family_col], errors="raise"
    ).astype("Int64")

    df_a[time_col] = pd.to_datetime(df_a[time_col])
    df_b[time_col] = pd.to_datetime(df_b[time_col])
    ref_df[time_col] = pd.to_datetime(ref_df[time_col])

    # ------------------------------------------------------------
    # Build lookup from reference:
    # [time, old_feature_family_id] -> new feature_family_id
    # Ignore invalid -1 rows
    # ------------------------------------------------------------
    lookup = ref_df.loc[
        ref_df[old_family_col].notna() & (ref_df[old_family_col] != -1),
        [time_col, old_family_col, family_col],
    ].copy()

    # Each (time, old_family) must map to exactly one new family ID
    check = lookup.groupby([time_col, old_family_col])[family_col].nunique(dropna=False)
    bad_keys = check[check > 1]
    if not bad_keys.empty:
        bad = lookup.merge(
            bad_keys.rename("__n").reset_index()[[time_col, old_family_col]],
            on=[time_col, old_family_col],
            how="inner",
        ).sort_values([time_col, old_family_col])

        raise ValueError(
            "Reference dataframe does not provide a unique mapping from "
            f"[{time_col}, {old_family_col}] to {family_col}.\n"
            f"Problematic rows:\n{bad.head(20)}"
        )

    lookup = lookup.drop_duplicates(subset=[time_col, old_family_col]).rename(
        columns={family_col: "__new_family_id"}
    )

    # ------------------------------------------------------------
    # Reassign df_b family IDs from reference
    # ------------------------------------------------------------
    df_b = df_b.drop(columns=[family_col], errors="ignore")

    df_b = df_b.merge(
        lookup,
        on=[time_col, old_family_col],
        how="left",
        sort=False,
    )

    df_b[family_col] = df_b["__new_family_id"]
    df_b = df_b.drop(columns="__new_family_id")

    # Invalid families stay -1
    invalid_minus1 = df_b[old_family_col] == -1
    df_b.loc[invalid_minus1, family_col] = -1

    # ------------------------------------------------------------
    # Fail loudly if any valid rows did not match
    # ------------------------------------------------------------
    unmatched = df_b[family_col].isna() & (df_b[old_family_col] != -1)
    if unmatched.any():
        bad = df_b.loc[unmatched, [time_col, old_family_col]].drop_duplicates().head(20)

        raise ValueError(
            f"Some valid rows in df_b could not be matched to ref_df, which would "
            f"produce NaN values in '{family_col}'.\n"
            f"Example unmatched keys:\n{bad}"
        )

    # ------------------------------------------------------------
    # Finalize and concatenate
    # ------------------------------------------------------------
    df_b[family_col] = df_b[family_col].astype("Int64")
    out = pd.concat([df_a, df_b], ignore_index=True)

    if out[family_col].isna().any():
        raise ValueError(f"Output contains NaN values in '{family_col}'.")

    out[family_col] = out[family_col].astype("Int64")
    out[old_family_col] = out[old_family_col].astype("Int64")

    return out


def concat_and_assign_family_ids_grid_from_reference_fast(
    da1,
    da2,
    ref_df,
    time_dim="time",
    family_col="feature_family_id",
    old_family_col="old_feature_family_id",
    invalid_values=(-1,),
):
    """
    Faster version of concat_and_assign_family_ids_grid_from_reference.

    It preserves the same logic and error behavior as closely as possible:
    - da1 is unchanged
    - da2 is remapped using ref_df on [time, old_feature_family_id]
    - invalid_values are preserved
    - if a time has no mapping but da2 contains valid IDs, raise
    - if a valid grid ID is missing from that time's mapping, raise
    - concatenates da1 and remapped da2 along time_dim
    """

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------
    if time_dim not in da1.dims:
        raise KeyError(f"'{time_dim}' not found in da1 dims.")
    if time_dim not in da2.dims:
        raise KeyError(f"'{time_dim}' not found in da2 dims.")
    if time_dim not in ref_df.columns:
        raise KeyError(f"'{time_dim}' not found in ref_df.")
    if old_family_col not in ref_df.columns:
        raise KeyError(f"'{old_family_col}' not found in ref_df.")
    if family_col not in ref_df.columns:
        raise KeyError(f"'{family_col}' not found in ref_df.")

    ref_df = ref_df.copy()
    ref_df[time_dim] = pd.to_datetime(ref_df[time_dim])
    ref_df[old_family_col] = pd.to_numeric(
        ref_df[old_family_col], errors="raise"
    ).astype("Int64")
    ref_df[family_col] = pd.to_numeric(ref_df[family_col], errors="raise").astype(
        "Int64"
    )

    # ------------------------------------------------------------
    # Build lookup table exactly like before
    # ------------------------------------------------------------
    lookup = ref_df.loc[
        ref_df[old_family_col].notna() & ref_df[family_col].notna(),
        [time_dim, old_family_col, family_col],
    ].copy()

    if invalid_values:
        lookup = lookup.loc[~lookup[old_family_col].isin(list(invalid_values))]

    dup_check = lookup.groupby([time_dim, old_family_col])[family_col].nunique(
        dropna=False
    )
    bad = dup_check[dup_check > 1]
    if not bad.empty:
        raise ValueError(
            f"Reference dataframe does not provide a unique mapping from "
            f"[{time_dim}, {old_family_col}] to {family_col}."
        )

    # time -> (sorted_old_ids, sorted_new_ids)
    time_to_arrays = {}
    for t, grp in lookup.groupby(time_dim, sort=False):
        old_ids = grp[old_family_col].astype(np.int64).to_numpy()
        new_ids = grp[family_col].astype(np.int64).to_numpy()

        order = np.argsort(old_ids)
        old_ids = old_ids[order]
        new_ids = new_ids[order]

        time_to_arrays[pd.Timestamp(t)] = (old_ids, new_ids)

    # ------------------------------------------------------------
    # Pull da2 data once, operate in-place on a copy
    # ------------------------------------------------------------
    da2_times = pd.to_datetime(da2[time_dim].values)

    data = np.array(da2.values, copy=True)
    time_axis = da2.get_axis_num(time_dim)
    data_t = np.moveaxis(data, time_axis, 0)  # time first view

    invalid_values = tuple(invalid_values)

    def valid_mask(arr):
        if not invalid_values:
            return np.ones(arr.shape, dtype=bool)
        if len(invalid_values) == 1:
            return arr != invalid_values[0]
        return ~np.isin(arr, invalid_values)

    # ------------------------------------------------------------
    # Remap per time slice
    # ------------------------------------------------------------
    for i, t in enumerate(da2_times):
        t = pd.Timestamp(t)
        arr = data_t[i]  # view into data
        mask = valid_mask(arr)

        mapping = time_to_arrays.get(t, None)

        if mapping is None:
            unique_valid = np.unique(arr[mask])
            if unique_valid.size > 0:
                raise ValueError(
                    f"No mapping found in ref_df for time {t}, but da2 contains "
                    f"valid family IDs such as {unique_valid[:10]}."
                )
            continue

        if not np.any(mask):
            continue

        old_ids, new_ids = mapping

        vals = arr[mask]
        unique_vals = np.unique(vals)

        # same fail-loudly behavior
        missing_mask = ~np.isin(unique_vals, old_ids, assume_unique=True)
        if np.any(missing_mask):
            missing = unique_vals[missing_mask]
            raise ValueError(
                f"At time {t}, some grid family IDs in da2 are not present in "
                f"ref_df mapping: {missing[:10].tolist()}"
            )

        # vectorized remap
        idx = np.searchsorted(old_ids, vals)
        arr[mask] = new_ids[idx]

    # ------------------------------------------------------------
    # Rebuild xarray only once
    # ------------------------------------------------------------
    da2_renumbered = xr.DataArray(
        data,
        coords=da2.coords,
        dims=da2.dims,
        attrs=da2.attrs,
        name=da2.name,
    )

    out = xr.concat([da1, da2_renumbered], dim=time_dim)
    return out


def to_cftime(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
    calendar: str,
    align_on: str = "date",
) -> cftime.datetime:
    """Converts a provided datetime-like object to a cftime datetime with the
    given calendar

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted
    calendar : str
        The requested cftime calender
    align_on : str, optional
        The 'align-on' parameter required for 360-day, 365-day and 366-day
        cftime dates, by default "date"

    Returns
    -------
    cftime.datetime
        A cftime object or array of cftime objects in the requested calendar
    """
    dates_arr = np.atleast_1d(dates)
    if isinstance(dates_arr[0], cftime.datetime):
        cftime_dates = (
            xr.DataArray(dates_arr, {"time": dates_arr})
            .convert_calendar(calendar, use_cftime=True, align_on=align_on)
            .time.values
        )
    else:
        cftime_dates = (
            xr.DataArray(dates_arr, {"time": pd.to_datetime(dates_arr)})
            .convert_calendar(calendar, use_cftime=True, align_on=align_on)
            .time.values
        )
    if not hasattr(dates, "__iter__") or isinstance(dates, str) and len(cftime_dates):
        return cftime_dates[0]
    return cftime_dates


def to_timestamp(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> pd.Timestamp:
    """Converts a provided datetime-like object to a pandas timestamp

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted

    Returns
    -------
    pd.Timestamp
        A pandas timestamp or array of pandas timestamps
    """
    squeeze_output = False
    if not hasattr(dates, "__iter__") or isinstance(dates, str):
        dates = np.atleast_1d(dates)
        squeeze_output = True

    if isinstance(next(iter(dates)), cftime.datetime):
        pd_dates = xr.CFTimeIndex(dates).to_datetimeindex()
    else:
        pd_dates = pd.to_datetime(dates)

    if squeeze_output:
        return next(iter(pd_dates))
    return pd_dates


def to_datetime(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> datetime.datetime:
    """Converts a provided datetime-like object to python datetime objects

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted

    Returns
    -------
    datetime.datetime
        A python datetime or array of python datetimes
    """
    return to_timestamp(dates).to_pydatetime()


def to_datetime64(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> np.datetime64:
    """Converts a provided datetime-like object to numpy datetime64 objects

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted

    Returns
    -------
    np.datetime64
        A numpy datetime64 or array of numpy datetime64s
    """
    return to_timestamp(dates).to_numpy()


def to_datestr(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> str:
    """Converts a provided datetime-like object to ISO format date strings

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted

    Returns
    -------
    str
        A string or array of strings in ISO date format
    """
    dates = to_datetime64(dates)
    if hasattr(dates, "__iter__"):
        return dates.astype(str)
    return str(dates)


def match_datetime_format(
    dates: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
    target: Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime],
) -> Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]:
    """Converts the provided datetime-like objects to the same datetime format
    as the provided target

    Parameters
    ----------
    dates : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects to be converted
    target : Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        A datetime-like object or array of datetime-like objects which the dates
        input will be converted to match

    Returns
    -------
    Union[str, datetime.datetime, np.datetime64, pd.Timestamp, cftime.datetime]
        The datetime-like values of the date parameter, converted to a format
        which matches that of the target input

    Raises
    ------
    ValueError
        If the target parameter provided is not a datetime-time object or array
        of datetime-like objects
    """
    if isinstance(target, str):
        return to_datestr(dates)
    if isinstance(target, xr.DataArray):
        target = target.values
    if isinstance(target, pd.Series):
        target = target.to_numpy()
    if hasattr(target, "__iter__"):
        target = target[0]
    if isinstance(target, str):
        return to_datestr(dates)
    if isinstance(target, cftime.datetime):
        return to_cftime(dates, target.calendar)
    if isinstance(target, pd.Timestamp):
        return to_timestamp(dates)
    if isinstance(target, np.datetime64):
        return to_datetime64(dates)
    if isinstance(target, datetime.datetime):
        return to_datetime(dates)
    raise ValueError("Target is not a valid datetime format")


def label_with_pbcs(
    in_label_arr: np.typing.ArrayLike,
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    connectivity: int = 2,
):
    """Function to run labeling, with checks for periodic boundaries.

    Parameters
    ----------
    in_label_arr: ArrayLike (bool)
        Input array to label. Can be 2D or 3D, but needs to be a binary
    PBC_flag: {"none", "hdim_1", "hdim_2", "both"}
        Flag to indicate which boundaries are periodic
    connectivity: int
        What kind of connectivity to use - the sklearn default is 2 (meaning diagonals are included).


    Returns
    -------
    ArrayLike (int)
        A labeled field
    """

    is_3d = len(np.shape(in_label_arr)) == 3

    labels, num_labels = skimage.measure.label(
        in_label_arr, background=0, return_num=True, connectivity=connectivity
    )
    if not is_3d:
        # let's transpose labels to a 1,y,x array to make calculations etc easier.
        labels = labels[np.newaxis, :, :]
    # these are [min, max], meaning that the max value is inclusive and a valid
    # value.
    z_min = 0
    z_max = labels.shape[0] - 1
    y_min = 0
    y_max = labels.shape[1] - 1
    x_min = 0
    x_max = labels.shape[2] - 1

    # deal with PBCs
    # all options that involve dealing with periodic boundaries
    pbc_options = ["hdim_1", "hdim_2", "both"]
    if PBC_flag not in pbc_options and PBC_flag != "none":
        raise ValueError(
            "Options for periodic are currently: none, " + ", ".join(pbc_options)
        )

    # we need to deal with PBCs in some way.
    if PBC_flag in pbc_options and num_labels > 0:
        #
        # create our copy of `labels` to edit
        labels_2 = copy.deepcopy(labels)
        # points we've already edited
        skip_list = np.array([])
        # labels that touch the PBC walls
        wall_labels = np.array([], dtype=np.int32)

        all_label_props = label_props.get_label_props_in_dict(labels)
        [
            all_labels_max_size,
            all_label_locs_v,
            all_label_locs_h1,
            all_label_locs_h2,
        ] = label_props.get_indices_of_labels_from_reg_prop_dict(all_label_props)

        # find the points along the boundaries

        # along hdim_1 or both horizontal boundaries
        if PBC_flag == "hdim_1" or PBC_flag == "both":
            # north and south wall
            ns_wall = np.unique(labels[:, (y_min, y_max), :])
            wall_labels = np.append(wall_labels, ns_wall)

        # along hdim_2 or both horizontal boundaries
        if PBC_flag == "hdim_2" or PBC_flag == "both":
            # east/west wall
            ew_wall = np.unique(labels[:, :, (x_min, x_max)])
            wall_labels = np.append(wall_labels, ew_wall)

        wall_labels = np.unique(wall_labels)

        for label_ind in wall_labels:
            new_label_ind = label_ind
            # 0 isn't a real index
            if label_ind == 0:
                continue
            # skip this label if we have already dealt with it.
            if np.any(label_ind == skip_list):
                continue

            # create list for skip labels for this wall label only
            skip_list_thisind = list()

            # get all locations of this label.
            label_locs_v = all_label_locs_v[label_ind]
            label_locs_h1 = all_label_locs_h1[label_ind]
            label_locs_h2 = all_label_locs_h2[label_ind]

            # loop through every point in the label
            for label_z, label_y, label_x in zip(
                label_locs_v, label_locs_h1, label_locs_h2
            ):
                # check if this is the special case of being a corner point.
                # if it's doubly periodic AND on both x and y boundaries, it's a corner point
                # and we have to look at the other corner.
                # here, we will only look at the corner point and let the below deal with x/y only.
                if PBC_flag == "both" and (
                    np.any(label_y == [y_min, y_max])
                    and np.any(label_x == [x_min, x_max])
                ):
                    # adjust x and y points to the other side
                    y_val_alt = pbc_utils.adjust_pbc_point(label_y, y_min, y_max)
                    x_val_alt = pbc_utils.adjust_pbc_point(label_x, x_min, x_max)

                    label_on_corner = labels[label_z, y_val_alt, x_val_alt]

                    if (label_on_corner != 0) and (
                        ~np.any(label_on_corner == skip_list)
                    ):
                        # alt_inds = np.where(labels==alt_label_3)
                        # get a list of indices where the label on the corner is so we can switch
                        # them in the new list.

                        labels_2[
                            all_label_locs_v[label_on_corner],
                            all_label_locs_h1[label_on_corner],
                            all_label_locs_h2[label_on_corner],
                        ] = label_ind
                        skip_list = np.append(skip_list, label_on_corner)
                        skip_list_thisind = np.append(
                            skip_list_thisind, label_on_corner
                        )

                    # if it's labeled and has already been dealt with for this label
                    elif (
                        (label_on_corner != 0)
                        and (np.any(label_on_corner == skip_list))
                        and (np.any(label_on_corner == skip_list_thisind))
                    ):
                        # print("skip_list_thisind label - has already been treated this index")
                        continue

                    # if it's labeled and has already been dealt with via a previous label
                    elif (
                        (label_on_corner != 0)
                        and (np.any(label_on_corner == skip_list))
                        and (~np.any(label_on_corner == skip_list_thisind))
                    ):
                        # find the updated label, and overwrite all of label_ind indices with
                        # updated label
                        labels_2_alt = labels_2[label_z, y_val_alt, x_val_alt]
                        labels_2[label_locs_v, label_locs_h1, label_locs_h2] = (
                            labels_2_alt
                        )
                        skip_list = np.append(skip_list, label_ind)
                        break

                # on the hdim1 boundary and periodic on hdim1
                if (PBC_flag == "hdim_1" or PBC_flag == "both") and np.any(
                    label_y == [y_min, y_max]
                ):
                    y_val_alt = pbc_utils.adjust_pbc_point(label_y, y_min, y_max)

                    # get the label value on the opposite side
                    label_alt = labels[label_z, y_val_alt, label_x]

                    # if it's labeled and not already been dealt with
                    if (label_alt != 0) and (~np.any(label_alt == skip_list)):
                        # find the indices where it has the label value on opposite side and change
                        # their value to original side
                        # print(all_label_locs_v[label_alt], alt_inds[0])
                        labels_2[
                            all_label_locs_v[label_alt],
                            all_label_locs_h1[label_alt],
                            all_label_locs_h2[label_alt],
                        ] = new_label_ind
                        # we have already dealt with this label.
                        skip_list = np.append(skip_list, label_alt)
                        skip_list_thisind = np.append(skip_list_thisind, label_alt)

                    # if it's labeled and has already been dealt with for this label
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (np.any(label_alt == skip_list_thisind))
                    ):
                        continue

                    # if it's labeled and has already been dealt with
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (~np.any(label_alt == skip_list_thisind))
                    ):
                        # find the updated label, and overwrite all of label_ind indices with
                        # updated label
                        labels_2_alt = labels_2[label_z, y_val_alt, label_x]
                        labels_2[label_locs_v, label_locs_h1, label_locs_h2] = (
                            labels_2_alt
                        )
                        new_label_ind = labels_2_alt
                        skip_list = np.append(skip_list, label_ind)

                if (PBC_flag == "hdim_2" or PBC_flag == "both") and (
                    np.any(label_x == x_min) or np.any(label_x == x_max)
                ):
                    x_val_alt = pbc_utils.adjust_pbc_point(label_x, x_min, x_max)

                    # get the label value on the opposite side
                    label_alt = labels[label_z, label_y, x_val_alt]

                    # if it's labeled and not already been dealt with
                    if (label_alt != 0) and (~np.any(label_alt == skip_list)):
                        # find the indices where it has the label value on opposite side and change
                        # their value to original side
                        labels_2[
                            all_label_locs_v[label_alt],
                            all_label_locs_h1[label_alt],
                            all_label_locs_h2[label_alt],
                        ] = new_label_ind
                        # we have already dealt with this label.
                        skip_list = np.append(skip_list, label_alt)
                        skip_list_thisind = np.append(skip_list_thisind, label_alt)

                    # if it's labeled and has already been dealt with for this label
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (np.any(label_alt == skip_list_thisind))
                    ):
                        continue

                    # if it's labeled and has already been dealt with
                    elif (
                        (label_alt != 0)
                        and (np.any(label_alt == skip_list))
                        and (~np.any(label_alt == skip_list_thisind))
                    ):
                        # find the updated label, and overwrite all of label_ind indices with
                        # updated label
                        labels_2_alt = labels_2[label_z, label_y, x_val_alt]
                        labels_2[label_locs_v, label_locs_h1, label_locs_h2] = (
                            labels_2_alt
                        )
                        new_label_ind = labels_2_alt
                        skip_list = np.append(skip_list, label_ind)

        # remove skipped labels from the number, remove 0 from the unique list.
        num_labels = len(np.unique(labels_2)) - 1
        return labels_2, num_labels
    else:
        return labels, num_labels


def find_df_rows_at_time(
    in_df: pd.DataFrame,
    in_time,
    time_var_name="time",
    time_padding: Optional[datetime.timedelta] = None,
):
    all_times = pd.Series(
        match_datetime_format(in_df[time_var_name], in_time),
        index=in_df.index,
    )

    if time_padding is not None:
        # padded_conv = pd.Timedelta(time_padding).to_timedelta64()
        if isinstance(in_time, (int, np.datetime64)):
            min_time = in_time - pd.Timedelta(time_padding).to_timedelta64()
            max_time = in_time + pd.Timedelta(time_padding).to_timedelta64()
        else:
            min_time = in_time - time_padding
            max_time = in_time + time_padding
        features_i = in_df.loc[all_times.between(min_time, max_time)]
    else:
        features_i = in_df.loc[all_times == in_time]
    return features_i


def identify_feature_families_from_data(
    feature_df: pd.DataFrame,
    in_data: xr.DataArray,
    threshold: float,
    return_grid: bool = False,
    family_column_name: str = "feature_family_id",
    time_padding: Optional[datetime.timedelta] = datetime.timedelta(seconds=0.5),
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    target: Literal["minimum", "maximum", "bool"] = "maximum",
    unlinked_family_id: Union[int, None] = -1,
):
    """
    Function to identify families/storm systems by identifying where segmentation touches.
    At a given time, segmentation areas are considered part of the same family if they
    touch at any point.

    Parameters
    ----------
    feature_df: pd.DataFrame
        Input feature dataframe
    in_data: xr.DataArray
        Input data. Should match the data that feature_df was generated from.
    threshold: float
        Threshold to define your feature family at
    return_grid: bool
        Whether to return the segmentation grid showing families
    family_column_name: str
        The name in the output dataframe of the family ID
    time_padding: datetime.timedelta
        Time padding to find the matching time between the feature_df and the in_data.
        By default, this is a half second to deal with random errors around time data type
        conversions.
    PBC_flag: {"none", "hdim_1", "hdim_2", "both"}
        What axes to do periodic boundaries on
    target: {"minimum", "maximum", "bool"}
        Whether we are looking for things ascending ("maximum") or descending ("minimum").
        There is the special case where you already have a true/false array, then you can put
        "bool" as the output.
    unlinked_family_id: int or None
        The value to have in the dataframe for any feature that cannot be linked to a family.
        This is unusual (as every feature should link to a family), but this can happen
        if e.g., the feature position is located outside of the feature area above the threshold.
        If "None", these features are dropped from the output.

    Returns
    -------
    pd.DataFrame and xr.DataArray or pd.DataFrame
        Input dataframe with family IDs associated with each feature
        if return_grid is True, the segmentation grid showing families is
        also returned.

    """

    # we need to label the data, but we currently label using skimage label, not dask label.

    time_var_name = "time"

    # 3D should be 4-D (time, then 3 spatial).
    # 2D should be 3-D (time, then 2 spatial)
    is_3D = len(in_data.shape) == 4

    seg_family_dict = dict()
    out_families = copy.deepcopy(in_data)
    out_families = out_families.astype(np.int64)
    out_families.name = "family_grid"
    max_family_number = 0
    enable_family_statistics = False

    if enable_family_statistics:
        region_props_vals = ["bbox", "centroid", "num_pixels"]
        family_stats = dict()
    for time_index in range(in_data.shape[0]):
        # TODO: fix time_var_name for isel?
        # print("time_index: ", time_index)
        in_data_at_time = in_data.isel(time=time_index)
        in_arr = np.array(in_data_at_time)

        # These are our families
        if target == "minimum":
            mask = in_arr < threshold
        elif target == "maximum":
            mask = in_arr > threshold
        elif target == "bool":
            mask = in_arr
        else:
            raise ValueError("target must be minimum, maximum, or bool")
        family_labeled_data, number_families = label_with_pbcs(
            mask, PBC_flag=PBC_flag, connectivity=1
        )
        if not is_3D:
            family_labeled_data = family_labeled_data[0]
        if enable_family_statistics:
            all_family_nums = list()

            family_props = skimage.measure.regionprops(family_labeled_data)
            # print(max_family_number)
            for family in family_props:
                all_family_nums.append(family.label + max_family_number)
                family_stats[family.label + max_family_number] = dict()
                # family_stats[family.label+max_family_number]['frame'] =

                family_stats[family.label + max_family_number][
                    "num_pixels"
                ] = family.area
                family_stats[family.label + max_family_number][family_column_name] = (
                    family.label + max_family_number
                )
                if not is_3D:
                    family_stats[family.label + max_family_number]["hdim_1_center"] = (
                        family.centroid[0]
                    )
                    family_stats[family.label + max_family_number]["hdim_2_center"] = (
                        family.centroid[1]
                    )

                else:
                    # TODO: integrate 3D stats - mostly around center coordinates - need the functions in tobac proper
                    raise NotImplementedError("3D stats not implemented yet")

        # need to associate family ID with each feature ID

        # get rows at current time

        rows_at_time = find_df_rows_at_time(
            feature_df,
            in_data_at_time["time"].values,
            time_var_name=time_var_name,
            time_padding=time_padding,
        )
        rows_at_time = rows_at_time.copy()

        if is_3D:
            v_max, h1_max, h2_max = family_labeled_data.shape
        else:
            # print(family_labeled_data.shape)
            h1_max, h2_max = family_labeled_data.shape

        rows_at_time["hdim_1_adj"] = np.clip(
            (rows_at_time["hdim_1"] + 0.5).astype(int), a_min=0, a_max=h1_max - 1
        )
        rows_at_time["hdim_2_adj"] = np.clip(
            (rows_at_time["hdim_2"] + 0.5).astype(int), a_min=0, a_max=h2_max - 1
        )

        data_in_shape = family_labeled_data.shape
        # TODO: deal with dim order for 3D
        if is_3D:
            if "vdim" not in rows_at_time:
                raise NotImplementedError(
                    "Family ID from raw field not supported going from 2D features to 3D family"
                )

            rows_at_time["vdim_adj"] = np.clip(
                (rows_at_time["vdim"] + 0.5).astype(int), a_min=0, a_max=v_max - 1
            )
            points_list = (
                rows_at_time["vdim_adj"].values,
                rows_at_time["hdim_1_adj"].values,
                rows_at_time["hdim_2_adj"].values,
            )

        else:
            points_list = (
                rows_at_time["hdim_1_adj"].values,
                rows_at_time["hdim_2_adj"].values,
            )

        #print(family_labeled_data.shape)

        #TODO BEE: instead of extracting the specific updraft center point here, should check for an overlap with an input mask (from W watershedding) that I give 
        # then assign each W feature a family (cloud) ID based on which cloud region the W region intersects with the most. 
        family_ids = family_labeled_data[points_list]

          
        # remove 0 (background) if needed
        family_ids_sorted = np.unique(family_ids[family_ids > 0])

        # we want to get rid of points that aren't features in the grid output
        suppressing_families = np.isin(family_labeled_data, family_ids_sorted)
        feature_id_family_id_match_ct = {
            feat: (fam_id + max_family_number if fam_id != 0 else -1)
            for feat, fam_id in zip(rows_at_time["feature"].values, family_ids)
        }
        seg_family_dict.update(feature_id_family_id_match_ct)
        out_families[time_index] = (
            family_labeled_data + max_family_number
        ) * suppressing_families.astype(int)

        max_family_number = out_families.max().values

    family_series = pd.Series(seg_family_dict, name=family_column_name)
    feature_series = pd.Series({x: x for x in seg_family_dict.keys()}, name="feature")
    family_df = pd.concat([family_series, feature_series], axis=1)
    out_df = feature_df.merge(family_df, on="feature", how="inner")
    if unlinked_family_id is not None:
        out_df.loc[out_df[family_column_name] == 0, family_column_name] = -1
    else:
        out_df = out_df[
            np.logical_and(
                out_df[family_column_name] != 0, out_df[family_column_name] != -1
            )
        ]

    if enable_family_statistics:
        family_stats_df = pd.DataFrame.from_dict(family_stats, orient="index")
        # we need to drop any family_stats that aren't in the feature DF
        family_stats_df = family_stats_df[
            family_stats_df[family_column_name].isin(
                np.unique(family_df[family_column_name].dropna().values)
            )
        ]
        fam_to_time_df = out_df[["time", family_column_name]].set_index(
            family_column_name
        )

        fam_to_time_df = fam_to_time_df.loc[
            ~fam_to_time_df.index.duplicated(keep="first"), :
        ].sort_index()
        fam_to_time_df = fam_to_time_df[fam_to_time_df.index != -1]
        family_stats_df = family_stats_df.join(
            fam_to_time_df, on=family_column_name
        ).dropna(subset="time")
        family_stats_df = family_stats_df.drop(
            [
                0,
            ],
            axis=0,
            errors="ignore",
        )
        family_stats_df = family_stats_df.set_index(family_column_name)

    if return_grid:
        if enable_family_statistics:
            return out_df, family_stats_df, out_families
        else:
            return out_df, out_families

    else:
        if enable_family_statistics:
            return out_df, family_stats_df
        else:
            return out_df


def combine_feature_families(
    in_feature_dfs: list[pd.DataFrame],
    in_stats: list[pd.DataFrame] = None,
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



    if in_stats is None:    
        enable_family_statistics = False

    if in_grid is not None:
        raise NotImplementedError("Merging of grids is not yet supported")

    # combine the feature dataframes
    combined_feature_df = tobac.utils.general.combine_feature_dataframes(
        in_feature_dfs,
        renumber_features=renumber_features,
        old_feature_column_name=old_feature_column_name,
    )
    # print(combined_feature_df)
    # get time: frame mapping
    time_frame_map = combined_feature_df.groupby("time")["frame"].max().to_dict()

    if enable_family_statistics:
        # now need to combine the stats dataframes
        combined_stats_df = pd.concat(in_stats)
        try:
            combined_stats_df["frame"] = [
                time_frame_map[x] for x in combined_stats_df["time"]
            ]
        except KeyError as e:
            return time_frame_map
        # print(combined_stats_df)

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

    
    if enable_family_statistics:
        combined_stats_df = combined_stats_df.reset_index().set_index(
            "feature_family_id", drop=False
        )

    if old_family_column_name is not None:
        combined_feature_df[old_family_column_name] = copy.deepcopy(
            combined_feature_df[family_column_name]
        )
        if enable_family_statistics:
            combined_stats_df[old_family_column_name] = copy.deepcopy(
                combined_stats_df[family_column_name]
            )

    new_family_numbers = np.empty(
        len(combined_feature_df[family_column_name]),
        dtype=combined_feature_df[family_column_name].dtype,
    )
    
    if enable_family_statistics:
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

    # for i, (index, row) in enumerate(combined_stats_df.iterrows()):
    #     curr_fam_pair = (row['frame'], row[family_column_name])
    #     new_family_numbers_stats[i] = family_number_map[curr_fam_pair]

    
    if enable_family_statistics:
        for i, (index, row) in enumerate(combined_stats_df.iterrows()):
            curr_fam_pair = (row["frame"], row[family_column_name])
            if curr_fam_pair in family_number_map:
                new_family_numbers_stats[i] = family_number_map[curr_fam_pair]
            else:
                print(f"Warning: family {curr_fam_pair} in stats not found in feature map")
                new_family_numbers_stats[i] = -1  # or np.nan if preferred

    combined_feature_df[family_column_name] = new_family_numbers
    
    
    if enable_family_statistics:
        combined_stats_df[family_column_name] = new_family_numbers_stats

        combined_stats_df = combined_stats_df.reset_index(drop=True).set_index(
            family_column_name, drop=False
        )
   
   
    if enable_family_statistics:
        return combined_feature_df, combined_stats_df
    else:
        return combined_feature_df

def track_feature_families(
    in_feat_arr: pd.DataFrame,
    in_family_stat_arr: pd.DataFrame,
    maintain_family_metric: Literal["cells", "area"] = "area",
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
        curr_time_feat_arr = reduced_feat_arr[reduced_feat_arr["frame"] == frame_num]

        # NEW: lookup the timestamp for this frame (assumes a single time per frame)
        curr_time = in_feat_arr.loc[
            in_feat_arr["frame"] == frame_num, time_col_name
        ].iloc[0]

        # get the relationship between feature family at this time -> cell at this time
        sets_curr_time = curr_time_feat_arr.groupby("feature_family_id")["cell"].agg(
            set
        )

        # first time we are dealing with
        if i == 0:
            # just set all feature families to the same tracked number
            sets_prev_time = copy.deepcopy(sets_curr_time)
            family_to_track_relationship = {
                sets_curr_time.index[tracked_num]: tracked_num + curr_tracked_family_num
                for tracked_num, _ in enumerate(sets_curr_time)
            }
            curr_tracked_family_num += len(sets_curr_time)
            continue

        # from the perspective of the previous time, what is the intersection between cells at t=0 and t=1?
        family_intersections_through_time_prev = dict()
        # from the perspective of the current time, what is the intersection between cells at t=0 and t=1?
        family_intersections_through_time_curr = dict()
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
                    if len(prev_curr_family_cell_overlap) > 1:
                        pass
                        # print(
                        #    "number of overlaps: ", len(prev_curr_family_cell_overlap)
                        # )

                    # record previous-> current links
                    if prev_family_label not in family_intersections_through_time_prev:
                        family_intersections_through_time_prev[prev_family_label] = [
                            (curr_family_label, prev_curr_family_cell_overlap),
                        ]
                    else:
                        family_intersections_through_time_prev[
                            prev_family_label
                        ].append((curr_family_label, prev_curr_family_cell_overlap))

                    # record current ->previous links
                    if curr_family_label not in family_intersections_through_time_curr:
                        family_intersections_through_time_curr[curr_family_label] = [
                            (prev_family_label, prev_curr_family_cell_overlap),
                        ]
                    else:
                        family_intersections_through_time_curr[
                            curr_family_label
                        ].append((prev_family_label, prev_curr_family_cell_overlap))

        # now that we have identified intersections, let's process them.
        for i_curr, curr_family_cells in enumerate(sets_curr_time):
            curr_family_label = sets_curr_time.index[i_curr]
            if (
                curr_family_label not in family_intersections_through_time_curr
                or len(family_intersections_through_time_curr[curr_family_label]) == 0
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

            elif len(family_intersections_through_time_curr[curr_family_label]) == 1:
                # we have only one family that shares cells from the previous time with the current time
                prev_family_label = family_intersections_through_time_curr[
                    curr_family_label
                ][0][0]
                # check if the previous family can only be linked to this cell.
                # print(prev_family_label, family_intersections_through_time_curr[curr_family_label])
                if len(family_intersections_through_time_prev[prev_family_label]) == 1:
                    family_to_track_relationship[curr_family_label] = (
                        family_to_track_relationship[prev_family_label]
                    )
                else:
                    # this family can only be linked to one previous cell, but the previous cell could be
                    # linked to multiple current families (split)
                    common_cells_prev_curr = family_intersections_through_time_curr[
                        curr_family_label
                    ][0][1]
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
                        # print("merge/split happening")
                        if curr_family_label in family_ids_of_interest:
                            print("Crashing out at too few cells")

                        # if it has too few cells from the previous family, it gets a new/unique number
                        family_to_track_relationship[curr_family_label] = (
                            curr_tracked_family_num
                        )
                        curr_tracked_family_num += 1
                        # log the split with time.
                        merge_split_list.append(
                            (
                                family_to_track_relationship[prev_family_label],
                                family_to_track_relationship[curr_family_label],
                                curr_time,
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
                    ) in family_intersections_through_time_curr[curr_family_label]:

                        keep_track_number = False
                        if maintain_family_metric == "cells":
                            # Link only if one parent has > maintain_track_cell_fraction of the child's cells
                            # this can only be triggered once
                            keep_track_number = (
                                len(parent_cell_intersection) / len(all_curr_cells)
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
                                        family_to_track_relationship[parent_cell_label]
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
                    # record the merger, regardless, with time.
                    for parent_cell_label, _ in family_intersections_through_time_curr[
                        curr_family_label
                    ]:
                        merge_split_list.append(
                            (
                                family_to_track_relationship[parent_cell_label],
                                family_to_track_relationship[curr_family_label],
                                curr_time,
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
                    ) in family_intersections_through_time_curr[curr_family_label]:

                        keep_track_number = False
                        if maintain_family_metric == "cells":
                            # Link only if one parent has > maintain_track_cell_fraction of the child's cells
                            # this can only be triggered once
                            keep_track_number = (
                                len(parent_cell_intersection) / len(all_curr_cells)
                            ) > maintain_track_cell_fraction and (
                                len(parent_cell_intersection) / len(all_prev_cells)
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
                                        family_to_track_relationship[parent_cell_label]
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
                            print("Crashing out at too few small area final", "")

                        family_to_track_relationship[curr_family_label] = (
                            curr_tracked_family_num
                        )
                        curr_tracked_family_num += 1

                    # record the merge/split mess with time
                    for parent_cell_label, _ in family_intersections_through_time_curr[
                        curr_family_label
                    ]:
                        merge_split_list.append(
                            (
                                family_to_track_relationship[parent_cell_label],
                                family_to_track_relationship[curr_family_label],
                                curr_time,
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

    return out_df, merge_split_list


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
        curr_frame_seg = in_family_track_df[in_family_track_df["time"] == curr_time]

        map_family_to_track_id = {x: 0 for x in np.unique(grid_at_time.values)}
        map_family_to_track_id.update(
            dict(
                curr_frame_seg.groupby([family_id_name, tracked_family_id_name])[
                    "frame"
                ]
                .count()
                .index.values
            )
        )

        return_grid.values[i] = vec_translate(
            return_grid.values[i], map_family_to_track_id
        )

    return_grid = return_grid.rename("tracked_family_id")

    return return_grid

def identify_feature_families_from_data_with_mask(
    feature_df: pd.DataFrame,
    feature_mask: xr.DataArray,
    in_data: xr.DataArray,
    threshold: float,
    return_grid: bool = False,
    family_column_name: str = "feature_family_id",
    time_padding: Optional[datetime.timedelta] = datetime.timedelta(seconds=0.5),
    PBC_flag: Literal["none", "hdim_1", "hdim_2", "both"] = "none",
    target: Literal["minimum", "maximum", "bool"] = "maximum",
    unlinked_family_id: Union[int, None] = -1,
    min_overlap_count: Union[int, None] = 64,
):
    """
    Function to identify families/storm systems by identifying where segmentation of in_data intersects at any point with an existing watershed mask.
    At a given time, segmentation areas are considered part of the same family if they
    touch at any point.

    Parameters
    ----------
    feature_df: pd.DataFrame
        Input feature dataframe
    feature_mask: xr.DataArray
        Input feature mask. Should match the data that feature_df was generated from.
    in_data: xr.DataArray
        Input data. Should match the data that feature_df was generated from.
    threshold: float
        Threshold to define your feature family at
    return_grid: bool
        Whether to return the segmentation grid showing families
    family_column_name: str
        The name in the output dataframe of the family ID
    time_padding: datetime.timedelta
        Time padding to find the matching time between the feature_df and the in_data.
        By default, this is a half second to deal with random errors around time data type
        conversions.
    PBC_flag: {"none", "hdim_1", "hdim_2", "both"}
        What axes to do periodic boundaries on
    target: {"minimum", "maximum", "bool"}
        Whether we are looking for things ascending ("maximum") or descending ("minimum").
        There is the special case where you already have a true/false array, then you can put
        "bool" as the output.
    unlinked_family_id: int or None
        The value to have in the dataframe for any feature that cannot be linked to a family.
        This is unusual (as every feature should link to a family), but this can happen
        if e.g., the feature position is located outside of the feature area above the threshold.
        If "None", these features are dropped from the output.
    min_overlap_count: int or None
        The minimum number of overlap points needed between input feature mask and mask from data

    Returns
    -------
    pd.DataFrame and xr.DataArray or pd.DataFrame
        Input dataframe with family IDs associated with each feature
        if return_grid is True, the segmentation grid showing families is
        also returned.

    """

    # we need to label the data, but we currently label using skimage label, not dask label.

    time_var_name = "time"

    # 3D should be 4-D (time, then 3 spatial).
    # 2D should be 3-D (time, then 2 spatial)
    is_3D = len(in_data.shape) == 4

    seg_family_dict = dict()
    seg_overlap_count_dict = dict()
    out_families = copy.deepcopy(in_data)
    out_families = out_families.astype(np.int64)
    out_families.name = "family_grid"
    max_family_number = 0
    enable_family_statistics = False

    if enable_family_statistics:
        region_props_vals = ["bbox", "centroid", "num_pixels"]
        family_stats = dict()
    for time_index in range(in_data.shape[0]):
        # TODO: fix time_var_name for isel?
        # print("time_index: ", time_index)
        in_data_at_time = in_data.isel(time=time_index)
        in_arr = np.array(in_data_at_time)

        # These are our families
        if target == "minimum":
            mask = in_arr < threshold
        elif target == "maximum":
            mask = in_arr > threshold
        elif target == "bool":
            mask = in_arr
        else:
            raise ValueError("target must be minimum, maximum, or bool")
        family_labeled_data, number_families = label_with_pbcs(
            mask, PBC_flag=PBC_flag, connectivity=1
        )
        if not is_3D:
            family_labeled_data = family_labeled_data[0]
        

        # need to associate family ID with each feature ID

        # get rows at current time

        rows_at_time = find_df_rows_at_time(
            feature_df,
            in_data_at_time["time"].values,
            time_var_name=time_var_name,
            time_padding=time_padding,
        )
        rows_at_time = rows_at_time.copy()

        #print(family_labeled_data.shape)

        # instead of extracting the specific updraft center point here, should check for an overlap with an input mask (from watershedding) 
        # then assign each feature a family ID based on which region the W region intersects with the most. 

        feature_mask_at_time = feature_mask.isel(time=time_index).values

        # find overlap of the family mask and original feature mask
        overlap_mask = (family_labeled_data>0) & (feature_mask_at_time>0)

        # get corresponding family and feature IDs for overlap points
        family_ids = family_labeled_data[overlap_mask]
        feature_ids =feature_mask_at_time[overlap_mask]

        # for each overlap point in space, assign the feature ID with associated family ID
        overlap_feature_family_ids = np.column_stack((feature_ids, family_ids))
        overlap_feature_family_ids, counts = np.unique(overlap_feature_family_ids, axis=0, return_counts=True)

        feature_id_family_id_match_ct = {}
        feature_id_overlap_count = {}
        max_counts = {}


        # for each feature ID, find which family ID has the largest overlapping region
        # and assign the feature ID to that family ID
        for (feat_id, fam_id), count in zip(overlap_feature_family_ids, counts):
            if (count > max_counts.get(feat_id,0)) & (count > min_overlap_count):

                max_counts[feat_id] = count
                feature_id_family_id_match_ct[feat_id] = fam_id+max_family_number
                feature_id_overlap_count[feat_id] = count

        seg_family_dict.update(feature_id_family_id_match_ct)
        seg_overlap_count_dict.update(feature_id_overlap_count)

        out_families[time_index] = (
            family_labeled_data + max_family_number
        )

        max_family_number = out_families.max().values

    family_series = pd.Series(seg_family_dict, name=family_column_name)
    feature_series = pd.Series({x: x for x in seg_family_dict.keys()}, name="feature")

    overlap_count_series = pd.Series(seg_overlap_count_dict, name=f"{family_column_name}_ncells_overlap")
    
    family_df = pd.concat([family_series, feature_series, overlap_count_series], axis=1)

    # note: as written, this currently drops any features that do not have overlap with any families
    out_df = feature_df.merge(family_df, on="feature", how="inner")

    if unlinked_family_id is not None:
        out_df.loc[out_df[family_column_name] == 0, family_column_name] = -1
    else:
        out_df = out_df[
            np.logical_and(
                out_df[family_column_name] != 0, out_df[family_column_name] != -1
            )
        ]

    if return_grid:
        if enable_family_statistics:
            return out_df, family_stats_df, out_families
        else:
            return out_df, out_families

    else:
        if enable_family_statistics:
            return out_df, family_stats_df
        else:
            return out_df


def renumber_feature_family_masks(in_feature_dfs, in_masks, 
    old_family_column_name: str = "feature_family_id_original",
    family_column_name: str = "feature_family_id",):
    """
    Function to renumber feature family masks to have unique feature family IDs across timesteps. 

    Parameters
    ----------
    in_feature_dfs: list[pd.DataFrame]
        List of dataframes generated by `identify_feature_families` then renumbered across times using `combine_feature_families`
    in_grid: list[xr.DataArray], optional
        List of DataArrays of the grids output by `identify_feature_families`.
    old_family_column_name: str or None, optional (default: "feature_family_id_original")
        The column name of old family numbers (corresponding to input mask)
    family_column_name: str
        The name in the output dataframe of the family ID
    """

    # build a lookup mapping between the old_id and new_id, where id_map[old_id] = new_id
    old_ids = in_feature_dfs[old_family_column_name].values.astype(int)
    new_ids = in_feature_dfs[family_column_name].values.astype(int)

    id_map = np.zeros(old_ids.max() +1, dtype=int)
    id_map[old_ids] = new_ids

    out_masks = xr.apply_ufunc(
        lambda arr: id_map[arr],
        in_masks,
        dask="allowed",
        vectorize=False
    )
    return out_masks