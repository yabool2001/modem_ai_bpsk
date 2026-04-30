import tomllib

with open ( "settings.toml" , "rb" ) as settings_toml_file :
    toml_settings = tomllib.load ( settings_toml_file )

BETA = float ( toml_settings[ "rrc_filter" ][ "BETA" ] )
SPAN = int ( toml_settings[ "rrc_filter" ][ "SPAN" ] )