/**
 * MSW browser worker setup (PRD_FRONTEND §2, §5).
 */

import { setupWorker } from "msw/browser";
import { handlers } from "./handlers";

export const worker = typeof window !== "undefined" ? setupWorker(...handlers) : null;
