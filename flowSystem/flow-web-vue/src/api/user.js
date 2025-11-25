import { get, post, requestInterceptor } from "@/assets/js/http";


// 用户注册
export async function register(data){
    try{
        const res = await post('/api/user/register', data)
        return res;
    }catch(error){
        throw new Error(`register error! : ${e}`);
    }
}

export async function updateUser(data){
    try{
        const res = await post('/api/user/update', data, requestInterceptor())
        console.log("updateUser:", res)
        return res;
    }catch(e){
        throw new Error(`updateUser error! : ${e}`);
    }
}

export async function deleteUser(data){
    try{
        const res = await post('/api/user/delete', data, requestInterceptor())
        console.log("deleteUser:", res)
        return res;
    }catch(e){
        throw new Error(`deleteUser error! : ${e}`);
    }
}

export async function fetchUserList(data){
    try{
        const res = await post('/api/user/list', data, requestInterceptor());
        return res;
    }catch(e){
        throw new Error(`fetchUserList error! : ${e}`);
    }
}

// ------------------------------------- 用户个人信息相关接口 ------------------------------------------ //
// 获取用户个人信息
export async function getUserProfile() {
    try {
        const res = await get('/api/user/profile', requestInterceptor())
        return res;
    } catch (error) {
        throw new Error(`获取用户信息失败: ${error}`);
    }
}

// 更新用户个人信息
export async function updateUserProfile(data) {
    try {
        const res = await post('/api/user/profile/update', data, requestInterceptor())
        return res;
    } catch (error) {
        throw new Error(`更新用户信息失败: ${error}`);
    }
}

// 修改密码
export async function changePassword(data) {
    try {
        const res = await post('/api/user/change-password', data, requestInterceptor())
        return res;
    } catch (error) {
        throw new Error(`修改密码失败: ${error}`);
    }
}

// 上传头像
export async function updateUserAvatar(data) {
    try {
        const res = await post('/api/user/update-avatar', data, 
            requestInterceptor(),
            'form-data'
        )
        return res;
    } catch (error) {
        throw new Error(`上传头像失败: ${error}`);
    }
}

export async function sendVerificationCode(data) {
    try {
        const res = await post('/api/user/forgot-password', data)
        return res;
    } catch (error) {
        throw new Error(`发送验证码失败: ${error}`);
    }
}
export async function resetPassword(data) {
    try {
        const res = await post('/api/user/reset-password', data)
        return res;
    } catch (error) {
        throw new Error(`重置密码失败: ${error}`);
    }
}