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
exports.HubdevicesEndpoint = void 0;
const endpoint_1 = require("../endpoint");
const endpoint_client_1 = require("../endpoint-client");
class HubdevicesEndpoint extends endpoint_1.Endpoint {
    constructor(config) {
        super(new endpoint_client_1.EndpointClient('hubdevices', config));
    }
    /**
     * Get a hub record
     * @param hubId
     */
    get(hubId) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.get(`${hubId}`);
        });
    }
    /**
     * Get the characteristics of a hub
     * @param hubId
     */
    getCharacteristics(hubId) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.get(`${hubId}/characteristics`);
        });
    }
    /**
     * Install driver on a hub. The primary use case of this functionality is to install a
     * self-published driver to be included in generic discovery (e.g. scanning).
     */
    installDriver(driverId, hubId, channelId) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.put(`${hubId}/drivers/${driverId}`, { channelId });
        });
    }
    /**
     * Change the driver for a device to the one specified by driverId.
     */
    switchDriver(driverId, hubId, deviceId, forceUpdate = false) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.patch(`${hubId}/childdevice/${deviceId}`, { driverId }, forceUpdate ? { forceUpdate: 'true' } : undefined);
        });
    }
    /**
     * Removes a driver from being installed on a hub. This will allow the hub to clean up the
     * deleted driver. However, all dependent devices need to be removed for cleanup to fully occur.
     */
    uninstallDriver(driverId, hubId) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.delete(`${hubId}/drivers/${driverId}`);
        });
    }
    /**
     * List drivers installed on the hub.
     *
     * @param deviceId When included, limit the drivers to those marked as matching the specified device.
     */
    listInstalled(hubId, deviceId) {
        return __awaiter(this, void 0, void 0, function* () {
            const params = deviceId ? { deviceId } : undefined;
            return this.client.get(`${hubId}/drivers`, params);
        });
    }
    getInstalled(hubId, driverId) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.get(`${hubId}/drivers/${driverId}`);
        });
    }
    /**
     * Returns the list of driver channels the hub is currently subscribed to.
     * Currently only returns the driver channel type.
     */
    enrolledChannels(hubId) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.get(`${hubId}/channels`, { channelType: 'DRIVERS' });
        });
    }
}
exports.HubdevicesEndpoint = HubdevicesEndpoint;
