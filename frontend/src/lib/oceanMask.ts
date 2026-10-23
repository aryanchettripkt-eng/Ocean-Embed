/**
 * North Indian Ocean geographical ocean mask for 40°E–100°E, 0°N–30°N.
 * Masks out terrestrial landmasses (India, Pakistan, Iran, Afghanistan, Arabia,
 * Bangladesh, Myanmar, Southeast Asia, Horn of Africa) so that ocean
 * temperature fields do not bleed over land (PRD_FRONTEND §2, §3.1).
 */
export function isNorthIndianOcean(lat: number, lon: number): boolean {
  if (lat < 0.0 || lat > 30.0 || lon < 40.0 || lon > 100.0) {
    return false;
  }

  // 1. Africa & Red Sea (west of 43°E, and Somalia / Horn of Africa)
  if (lon <= 43.0) {
    return false;
  }
  if (lon <= 51.5 && lat <= 12.0) {
    // Somalia / Kenya coast: (0, 42.0) to (12, 51.5)
    if (lon < 42.0 + 0.8 * lat) {
      return false;
    }
  }

  // 2. Arabian Peninsula (Saudi Arabia, Yemen, Oman, UAE)
  if (lon < 60.0 && lat > 12.0) {
    let coastLat: number;
    if (lon <= 48.0) {
      coastLat = 12.5 + (14.5 - 12.5) * (lon - 43.0) / 5.0;
    } else if (lon <= 55.0) {
      coastLat = 14.5 + (17.5 - 14.5) * (lon - 48.0) / 7.0;
    } else {
      coastLat = 17.5 + (22.5 - 17.5) * (lon - 55.0) / 5.0;
    }

    if (lat > coastLat) {
      // Allow Gulf of Oman entrance
      const isGulf = lat <= 26.0 && lon >= 56.5 && lon <= 60.0;
      if (!isGulf) {
        return false;
      }
    }
  }

  // 3. Everything north of 25.5°N east of 57°E is Land (Iran, Pakistan, Afghanistan, North India, Tibet)
  if (lat >= 25.5 && lon >= 57.0) {
    return false;
  }

  // 4. Pakistan & Gujarat coast
  if (lat >= 24.5 && lon >= 62.0 && lon <= 74.0) {
    return false;
  }
  if (lat >= 23.0 && lon >= 70.0 && lon <= 74.0) {
    return false;
  }

  // 5. Indian Subcontinent
  if (lat >= 8.0 && lat <= 24.0 && lon >= 69.0 && lon <= 88.0) {
    // West coast: 8°N at 77.5°E to 22°N at 69.5°E
    const lonW = 77.5 - ((77.5 - 69.5) * (lat - 8.0)) / (22.0 - 8.0);
    // East coast: 8°N at 77.5°E to 21.5°N at 87.0°E
    const lonE = 77.5 + ((87.0 - 77.5) * (lat - 8.0)) / (21.5 - 8.0);
    if (lon >= lonW && lon <= lonE) {
      return false;
    }
  }

  // Sri Lanka
  if (lat >= 5.8 && lat <= 9.8 && lon >= 79.5 && lon <= 82.0) {
    return false;
  }

  // 6. North of 22°N east of 69°E is Land (North India, Bangladesh, Myanmar, Tibet)
  if (lat >= 22.0 && lon >= 69.0) {
    return false;
  }

  // 7. Southeast Asia (Myanmar, Thailand, Malay Peninsula, Sumatra)
  if (lat >= 16.0 && lon >= 94.0) {
    return false;
  }
  if (lat >= 13.0 && lon >= 98.0) {
    return false;
  }
  if (lat >= 6.0 && lon >= 99.0) {
    return false;
  }
  if (lat <= 5.5 && lon >= 95.0) {
    if (lon > 95.0 + 0.9 * (5.5 - lat)) {
      return false;
    }
  }

  return true;
}
