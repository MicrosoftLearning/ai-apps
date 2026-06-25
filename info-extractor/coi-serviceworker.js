/*
 * COI service worker bootstrap for static hosting (including GitHub Pages).
 * This enables SharedArrayBuffer-dependent features by applying COOP/COEP.
 */
if (typeof window === "undefined") {
    self.addEventListener("install", () => self.skipWaiting());
    self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

    self.addEventListener("fetch", (event) => {
        const request = event.request;

        if (request.cache === "only-if-cached" && request.mode !== "same-origin") {
            return;
        }

        event.respondWith(
            fetch(request)
                .then((response) => {
                    if (response.status === 0) {
                        return response;
                    }

                    const headers = new Headers(response.headers);
                    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
                    headers.set("Cross-Origin-Opener-Policy", "same-origin");
                    headers.set("Cross-Origin-Resource-Policy", "cross-origin");

                    return new Response(response.body, {
                        status: response.status,
                        statusText: response.statusText,
                        headers,
                    });
                })
                .catch((error) => {
                    return new Response(String(error), {
                        status: 500,
                        statusText: "Service Worker Fetch Failed",
                    });
                })
        );
    });
} else {
    (async () => {
        if (!window.isSecureContext) {
            return;
        }

        if (!("serviceWorker" in navigator)) {
            return;
        }

        try {
            const workerUrl = new URL("coi-serviceworker.js", window.location.href).toString();
            const registration = await navigator.serviceWorker.register(workerUrl);

            // Reload once when a new worker takes control so COI headers become active.
            if (!navigator.serviceWorker.controller) {
                window.location.reload();
                return;
            }

            if (registration.waiting) {
                registration.waiting.postMessage({ type: "SKIP_WAITING" });
            }

            let reloaded = false;
            navigator.serviceWorker.addEventListener("controllerchange", () => {
                if (reloaded) {
                    return;
                }
                reloaded = true;
                window.location.reload();
            });
        } catch (_error) {
            // If service worker registration fails, app still loads without COI.
        }
    })();
}
