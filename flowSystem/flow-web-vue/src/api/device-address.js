import { post, requestInterceptor } from '@/assets/js/http';

// 获取地址列表
export async function getDeviceAddressList(data) {
    return post('/api/device/address/list', data, requestInterceptor());
}

// 新增地址
export async function addDeviceAddress(data) {
    return post('/api/device/address/add', data, requestInterceptor());
}
// 更新地址
export async function updateDeviceAddress(data) {
    return post('/api/device/address/update', data, requestInterceptor());
}

// 删除地址
export async function deleteDeviceAddress(data) {
    return post('/api/device/address/delete', data, requestInterceptor());
}

// 批量删除地址
export async function batchDeleteDeviceAddress(data) {
    return post('/api/device/address/batch-delete', data, requestInterceptor());
}

// 更新地址状态
export async function updateDeviceAddressStatus(data) {
    return post('/api/device/address/update-status', data, requestInterceptor());
}