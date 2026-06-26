
DIM_MAPPINGS = {
    "RAMS": {
        "phony_dim_3": "z",
        "phony_dim_1": "y",
        "phony_dim_2": "x",
    },
    "WRF": {
        'west_east': 'x',
        'south_north': 'y',
        'bottom_top': 'z',
        'west_east_stag': 'x_stag',
        'south_north_stag': 'y_stag',
        'bottom_top_stag': 'z_stag'
    }
}

VAR_MAPPINGS = {
    'RAMS': { # TODO: Considering unstaggering vertical velocity from the very beginning so it's easier to compare with condensate down the line
        'vertical_velocity': 'WP', #note: in RAMS, WP is the vertical velocity at this timestep (WC says "current" in docs but this is actually the unfiltered future value for next time step initial conditions; see the Subroutine predict for details)
        'zonal_velocity': 'UP',
        'meridional_velocity': 'VP',
        'cloud_condensate':(['RCP','RSP','RPP'], lambda ds: ds['RCP'] + ds['RSP'] + ds['RPP']),
        'total_condensate':(['RCP','RDP','RPP','RGP','RAP','RHP', 'RSP','RPP'], lambda ds: ds['RCP'] +ds['RDP'] + ds['RPP'] +ds['RGP'] + ds['RAP'] + ds['RHP'] + ds['RSP'] + ds['RPP']), 
    },
    'WRF': {
        'vertical_velocity': 'W', 
        'zonal_velocity': 'U',
        'meridional_velocity': 'V'
    }
}


DIMS = {
    'x': {
        'short_name': 'xdist', 
        'rams_header_name': '__xtn', 
        'long_name': 'E-W distance', 
        'standard_name': 'projection_x_coordinate', 
        'units': 'meters'
    },
    'y': {
        'short_name': 'ydist', 
        'rams_header_name': '__ytn', 
        'long_name': 'N-S distance', 
        'standard_name': 'projection_y_coordinate',  
        'units': 'meters'
    },
    'z': {
        'short_name': 'altitude', 
        'rams_header_name': '__ztn', 
        'long_name': 'altitude above ground level', 
        'standard_name': 'altitude', 
        'units': 'meters'
    },
    'x_stag': {
        'short_name': 'xdist_stag', 
        'rams_header_name': '__xmn', 
        'long_name': 'E-W distance', 
        'standard_name': 'projection_x_coordinate', 
        'units': 'meters'
    },
    'y_stag': {
        'short_name': 'ydist_stag', 
        'rams_header_name': '__ymn', 
        'long_name': 'N-S distance', 
        'standard_name': 'projection_y_coordinate', 
        'units': 'meters'
    },
    'z_stag': {
        'short_name': 'altitude_stag', 
        'rams_header_name': '__zmn', 
        'long_name': 'altitude above ground level', 
        'standard_name': 'altitude', 
        'units': 'meters'
    },
}

GRID_SPACING_MAPPINGS = {
    1: 1600.0,  # Grid 1 spacing 
    2: 400.0,   # Grid 2 spacing
    3: 100.0,   # Grid 3 spacing
}