"use strict";

let global = window;
let WORKER_ENABLED = !!(global === global.window && global.URL && global.Blob && global.Worker);

function InlineWorker(func, self) {
  var _this = this;
  var functionBody;
  self = self || {};

  if (WORKER_ENABLED) {
    functionBody = func.toString().trim().match(
      /^function\s*\w*\s*\([\w\s,]*\)\s*{([\w\W]*?)}$/
    )[1];
    return new global.Worker(global.URL.createObjectURL(new global.Blob([functionBody], { type: "text/javascript" })));
  }
}


