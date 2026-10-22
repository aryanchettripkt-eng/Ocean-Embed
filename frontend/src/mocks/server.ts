/**
 * MSW Node.js server setup for tests (PRD_FRONTEND §2, §5).
 */

import { setupServer } from "msw/node";
import { handlers } from "./handlers";

export const server = setupServer(...handlers);
