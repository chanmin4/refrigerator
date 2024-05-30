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
exports.ChannelsEndpoint = void 0;
const endpoint_1 = require("../endpoint");
const endpoint_client_1 = require("../endpoint-client");
class ChannelsEndpoint extends endpoint_1.Endpoint {
    constructor(config) {
        super(new endpoint_client_1.EndpointClient('distchannels', config));
    }
    create(data) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.post('', data);
        });
    }
    delete(id) {
        return __awaiter(this, void 0, void 0, function* () {
            yield this.client.delete(id);
        });
    }
    update(id, data) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.put(id, data);
        });
    }
    get(id) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.get(id);
        });
    }
    getDriverChannelMetaInfo(channelId, driverId) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.get(`${channelId}/drivers/${driverId}/meta`);
        });
    }
    list(options = {}) {
        return __awaiter(this, void 0, void 0, function* () {
            const params = {};
            if (options.subscriberType) {
                params.type = options.subscriberType;
            }
            if (options.subscriberId) {
                if (!options.subscriberType) {
                    throw Error('specifying a subscriberId requires also specifying a subscriberType');
                }
                params.subscriberId = options.subscriberId;
            }
            if (typeof (options.includeReadOnly) === 'boolean') {
                params.includeReadOnly = options.includeReadOnly.toString();
            }
            return this.client.getPagedItems('', params);
        });
    }
    listAssignedDrivers(channelId) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.getPagedItems(`${channelId}/drivers`);
        });
    }
    /**
     * Assign or publish a driver to a channel.
     *
     * NOTE: This method also works to update the driver version assigned to a channel.
     */
    assignDriver(channelId, driverId, version) {
        return __awaiter(this, void 0, void 0, function* () {
            return this.client.post(`${channelId}/drivers`, { driverId, version });
        });
    }
    unassignDriver(channelId, driverId) {
        return __awaiter(this, void 0, void 0, function* () {
            yield this.client.delete(`${channelId}/drivers/${driverId}`);
        });
    }
    enrollHub(channelId, hubId) {
        return __awaiter(this, void 0, void 0, function* () {
            yield this.client.post(`${channelId}/hubs/${hubId}`);
        });
    }
    unenrollHub(channelId, hubId) {
        return __awaiter(this, void 0, void 0, function* () {
            yield this.client.delete(`${channelId}/hubs/${hubId}`);
        });
    }
}
exports.ChannelsEndpoint = ChannelsEndpoint;
