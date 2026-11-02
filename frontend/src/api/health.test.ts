import { afterEach, expect, it, vi } from "vitest";

import { checkHealth } from "./health";

afterEach(() => vi.unstubAllGlobals());

const respond = (r: Response | Error) =>
  vi.stubGlobal("fetch", vi.fn(async () => { if (r instanceof Error) throw r; return r; }));

it("is ok when the API and database are ready", async () => {
  respond(new Response(JSON.stringify({ status: "ok", checks: { database: "ok" } }), { status: 200 }));
  expect(await checkHealth()).toBe("ok");
});

it("is degraded when the API answers but the database is down", async () => {
  respond(new Response(JSON.stringify({ status: "unavailable", checks: { database: "timeout" } }), { status: 503 }));
  expect(await checkHealth()).toBe("degraded");
});

it("is offline when nothing answers or a proxy does", async () => {
  respond(new TypeError("Failed to fetch"));
  expect(await checkHealth()).toBe("offline");
  respond(new Response("proxy error", { status: 500 }));
  expect(await checkHealth()).toBe("offline");
});