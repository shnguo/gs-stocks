import { spawn } from "node:child_process";

const environment = {
  ...process.env,
  MOOTDX_DATA_API_BASE_URL:
    process.env.MOOTDX_DATA_API_BASE_URL ?? "http://127.0.0.1:3101",
};
const localProxyPort = process.env.MOOTDX_LOCAL_PROXY_PORT ?? "3101";
const configuredRealtimeBase = process.env.MOOTDX_REALTIME_API_BASE_URL;
const localRealtimeUpstream = process.env.MOOTDX_LOCAL_REALTIME_UPSTREAM_URL ??
  (process.env.MOOTDX_REALTIME_TRANSPORT === "polling" ? configuredRealtimeBase : undefined);
const proxyEnvironment = localRealtimeUpstream
  ? {
      ...environment,
      MOOTDX_LOCAL_REALTIME_UPSTREAM_URL: localRealtimeUpstream,
      MOOTDX_LOCAL_REALTIME_POLL_INTERVAL_MS:
        process.env.MOOTDX_LOCAL_REALTIME_POLL_INTERVAL_MS ?? "1000",
    }
  : environment;
const dashboardEnvironment = localRealtimeUpstream
  ? {
      ...environment,
      MOOTDX_REALTIME_API_BASE_URL: `http://127.0.0.1:${localProxyPort}`,
      MOOTDX_REALTIME_TRANSPORT: "websocket",
    }
  : environment;

const children = [
  spawn(process.execPath, ["scripts/mootdx-preview-proxy.mjs"], {
    env: proxyEnvironment,
    stdio: "inherit",
  }),
  spawn("npm", ["run", "dev"], {
    env: dashboardEnvironment,
    stdio: "inherit",
  }),
];

let stopping = false;

function stop(signal = "SIGTERM") {
  if (stopping) return;
  stopping = true;
  for (const child of children) {
    if (child.exitCode === null) child.kill(signal);
  }
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => stop(signal));
}

for (const child of children) {
  child.on("error", (error) => {
    process.stderr.write(`${JSON.stringify({ event: "local-dev-child-error", error: error.message })}\n`);
    stop();
    process.exitCode = 1;
  });
  child.on("exit", (code, signal) => {
    if (stopping) return;
    process.stderr.write(`${JSON.stringify({ event: "local-dev-child-exit", code, signal })}\n`);
    stop();
    process.exitCode = code ?? 1;
  });
}
