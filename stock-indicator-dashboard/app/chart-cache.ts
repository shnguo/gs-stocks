"use client";

const DATABASE_NAME = "stock-indicator-dashboard";
const DATABASE_VERSION = 1;
const STORE_NAME = "chart-payloads";

type StoredChartPayload = {
  key: string;
  savedAt: string;
  payload: unknown;
};

export async function readChartCache(key: string): Promise<unknown | null> {
  if (typeof indexedDB === "undefined") return null;
  const database = await openDatabase();
  try {
    const record = await requestResult<StoredChartPayload | undefined>(
      database.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(key),
    );
    return record?.payload ?? null;
  } finally {
    database.close();
  }
}

export async function writeChartCache(key: string, payload: unknown): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  const database = await openDatabase();
  try {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).put({
      key,
      savedAt: new Date().toISOString(),
      payload,
    } satisfies StoredChartPayload);
    await transactionComplete(transaction);
  } finally {
    database.close();
  }
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.addEventListener("upgradeneeded", () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "key" });
      }
    });
    request.addEventListener("success", () => resolve(request.result), { once: true });
    request.addEventListener("error", () => reject(request.error), { once: true });
    request.addEventListener(
      "blocked",
      () => reject(new Error("chart cache database upgrade is blocked")),
      { once: true },
    );
  });
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.addEventListener("success", () => resolve(request.result), { once: true });
    request.addEventListener("error", () => reject(request.error), { once: true });
  });
}

function transactionComplete(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.addEventListener("complete", () => resolve(), { once: true });
    transaction.addEventListener("abort", () => reject(transaction.error), { once: true });
    transaction.addEventListener("error", () => reject(transaction.error), { once: true });
  });
}
