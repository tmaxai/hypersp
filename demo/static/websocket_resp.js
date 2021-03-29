"use strict";

var ws_flag = true
class Ws {

    get newClientPromise() {
        return new Promise((resolve, reject) => {
            let wsClient = new WebSocket("wss://127.0.0.1:15003/recognize");
            wsClient.onopen = () => {
                console.log("connected");
                resolve(wsClient);
            };
            wsClient.onerror = error => reject(error);
            wsClient.onmessage = function (message) {
                var result = JSON.parse(message["data"])
                if (result["text"]) {
                    textArea.value = result["text"];
                    if (result["final"]) {
                        wsClient.close()
                    }
                }
            }
            wsClient.onclose = function (evt) {
                console.log("closed");
                this.promise = null
                ws_flag = true
                console.log(ws_flag)
            }
        })
    }
    get clientPromise() {
        if (ws_flag) {
            this.promise = this.newClientPromise
            ws_flag = false
        }
        return this.promise
    }
}
