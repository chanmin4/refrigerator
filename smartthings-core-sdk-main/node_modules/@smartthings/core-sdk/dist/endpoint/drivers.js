"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DriversEndpoint = void 0;
const endpoint_1 = require("../endpoint");
const endpoint_client_1 = require("../endpoint-client");
class DriversEndpoint extends endpoint_1.Endpoint {
    constructor(config) {
        super(new endpoint_client_1.EndpointClient('drivers', config));
    }
    get(id) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.get(id);
        });
    }
    getRevision(id, version) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.get(`${id}/versions/${version}`);
        });
    }
    delete(id) {
        return __awaiter(this, void 0, void 0, function* () {
            yield this.client.delete(id);
        });
    }
    list() {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.getPagedItems('');
        });
    }
    /**
     * List drivers in the default channel. (The default channel in this context is a channel
     * that users do not need to subscribe to.)
     */
    listDefault() {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.getPagedItems('default');
        });
    }
    /**
     * Uploads the zipped package represented by archiveData.
     */
    upload(archiveData) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.request('post', 'package', archiveData, undefined, { headerOverrides: { 'Content-Type': 'application/zip' } });
        });
    }
}
exports.DriversEndpoint = DriversEndpoint;
