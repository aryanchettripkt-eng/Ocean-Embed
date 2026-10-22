"""Fixed caveat strings mandated by scientific governance (PRD_BACKEND Section 7)."""

GLORYS_CAVEAT: str = (
    "Predictions are trained to reproduce the GLORYS reanalysis. "
    "GLORYS assimilates satellite altimetry and ARGO profiles, so "
    "agreement with GLORYS is not the same as agreement with the real ocean."
)

ARGO_CAVEAT: str = (
    "ARGO profiles are also assimilated into GLORYS, so this comparison is not fully independent."
)

SYNTHETIC_CAVEAT: str = "This deployment is serving synthetic demo data, not model results."
