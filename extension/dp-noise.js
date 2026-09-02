/**
 * dp-noise.js
 *
 * Differential-privacy noise layer — PRD Sections 4/6/7: "inject
 * calibrated differential-privacy noise into the sanitized payload
 * before it leaves the client."
 *
 * IMPORTANT correctness note (read before touching this file):
 * Noise from this module must NEVER be applied to the coordinates used
 * to actually draw a redaction box in offscreen.js. Jittering a box
 * smaller or shifting it before the blackout is painted could uncover a
 * sliver of a real password/card number — an actual privacy regression,
 * not a hardening. Noise only ever touches (a) a *copy* of region
 * metadata built AFTER the blackout has already been drawn from the
 * exact, un-noised coordinates, and (b) pixels that are already outside
 * every redacted rect.
 *
 * What gets noised, and why:
 *   1. Region geometry (x/y/w/h of every redacted box, in the metadata
 *      sent to the server). The blacked-out pixels already hide the
 *      *content* of a password/card/OTP field, but its exact bounding
 *      box still leaks a little: field width can hint at expected input
 *      length, and precise repeated coordinates could fingerprint a
 *      page's exact layout. Laplace noise blurs this.
 *   2. Non-redacted pixel data, in coarse blocks rather than per-pixel
 *      (a full per-pixel pass over a 1080p frame would dwarf the rest of
 *      the pipeline's latency combined), to perturb any residual signal
 *      in the visible parts of the screenshot.
 *
 * Mechanism: standard Laplace mechanism, noise ~ Lap(0, b) where
 * b = sensitivity / epsilon. Smaller epsilon = more privacy, more noise.
 *
 * Epsilon is intentionally generous here — this is a supplementary
 * hardening layer on top of full redaction, not the only thing standing
 * between a password and the network. A very small epsilon would visibly
 * degrade the VLM's layout understanding for little added benefit, since
 * the actual sensitive content is already fully opaque either way.
 */

const DP_EPSILON_GEOMETRY = 2.0;   // region bbox coordinates (pixels)
const DP_EPSILON_PIXELS = 4.0;     // non-redacted pixel blocks (0-255 scale)
const GEOMETRY_SENSITIVITY = 8;    // px — calibration bound for coordinate shift
const PIXEL_SENSITIVITY = 12;      // 0-255 scale — calibration bound for channel shift
const PIXEL_BLOCK_SIZE = 16;       // noise applied per 16x16 block, not per pixel

/** Inverse-CDF sampling of Lap(0, scale). */
function sampleLaplace(scale) {
  const u = Math.random() - 0.5; // u in (-0.5, 0.5)
  const sign = u < 0 ? -1 : 1;
  const magnitude = Math.min(Math.abs(u) * 2, 1 - 1e-12); // keep ln() finite
  return -scale * sign * Math.log(1 - magnitude);
}

function clamp8(v) {
  return v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
}

/**
 * Returns a NEW array with Laplace-jittered rects. Never mutates the
 * input — callers must only feed this a copy taken after the real
 * blackout has already been drawn from the original, exact regions.
 */
function addGeometryNoise(regions, epsilon = DP_EPSILON_GEOMETRY) {
  const b = GEOMETRY_SENSITIVITY / epsilon;
  const jitter = (v) => Math.max(0, Math.round(v + sampleLaplace(b)));
  return regions.map((r) => {
    if (!r.rect) return { ...r };
    return {
      ...r,
      rect: {
        x: jitter(r.rect.x),
        y: jitter(r.rect.y),
        w: Math.max(1, jitter(r.rect.w)),
        h: Math.max(1, jitter(r.rect.h)),
      },
    };
  });
}

/**
 * Adds block-level Laplace noise to an ImageData's RGB channels,
 * skipping any block that overlaps an already-redacted rect (no point
 * noising a solid black box, and it keeps the fill color exact for the
 * VLM's redaction-aware prompt to recognize reliably). Mutates and
 * returns the same ImageData in place — call on a canvas copy, never on
 * the bitmap you still need pristine elsewhere.
 *
 * `exactRects` must be the real, un-noised redaction rects (device-pixel
 * scaled) — this is a safety check, not a place to reuse noised geometry.
 */
function addPixelNoise(imageData, exactRects, epsilon = DP_EPSILON_PIXELS) {
  const { data, width, height } = imageData;
  const b = PIXEL_SENSITIVITY / epsilon;

  const overlapsRedacted = (bx, by, size) => {
    for (const r of exactRects) {
      if (
        bx < r.x + r.w &&
        bx + size > r.x &&
        by < r.y + r.h &&
        by + size > r.y
      ) {
        return true;
      }
    }
    return false;
  };

  for (let by = 0; by < height; by += PIXEL_BLOCK_SIZE) {
    for (let bx = 0; bx < width; bx += PIXEL_BLOCK_SIZE) {
      if (overlapsRedacted(bx, by, PIXEL_BLOCK_SIZE)) continue;

      const dr = sampleLaplace(b);
      const dg = sampleLaplace(b);
      const db = sampleLaplace(b);
      const maxX = Math.min(bx + PIXEL_BLOCK_SIZE, width);
      const maxY = Math.min(by + PIXEL_BLOCK_SIZE, height);

      for (let y = by; y < maxY; y++) {
        for (let x = bx; x < maxX; x++) {
          const idx = (y * width + x) * 4;
          data[idx] = clamp8(data[idx] + dr);
          data[idx + 1] = clamp8(data[idx + 1] + dg);
          data[idx + 2] = clamp8(data[idx + 2] + db);
          // alpha channel left untouched
        }
      }
    }
  }
  return imageData;
}

self.SquintDP = {
  addGeometryNoise,
  addPixelNoise,
  DP_EPSILON_GEOMETRY,
  DP_EPSILON_PIXELS,
};
