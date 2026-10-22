"""Fixed caveat strings for OceanScope AI (PRD_BACKEND Section 7)."""

GLORYS_CAVEAT: str = (
    "Predictions are trained to reproduce the GLORYS reanalysis. "
    "GLORYS assimilates satellite altimetry and ARGO profiles, so agreement "
    "with GLORYS is not the same as agreement with the real ocean."
)

ARGO_CAVEAT: str = (
    "ARGO profiles are also assimilated into GLORYS, so this comparison is not fully independent."
)

LIVE_CAVEAT: str = (
    "Live maps are made about 3 days behind real time from near-real-time inputs. "
    "There is no GLORYS reference for these dates yet, so they have not been "
    "checked the way validation dates have."
)

HEADLINE_CRITERION: str = (
    "Bay of Bengal, 50–300 m: model RMSE vs held-out ARGO floats lower than "
    "GLORYS12, the GREP ensemble mean and ARMOR3D (95% float-bootstrap "
    "interval of the difference below zero)."
)

SYNTHETIC_CAVEAT: str = "This deployment is serving synthetic demo data, not model results."
