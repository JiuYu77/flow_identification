import { post, get, requestInterceptor } from '@/assets/js/http';

/**
 * 获取数据文件列表（支持层级目录）
 * @param {Object} data - 查询参数
 * @param {string} data.keyword - 搜索关键词
 * @param {string} data.fileType - 文件类型
 * @param {string} data.path - 目录路径
 * @param {number} data.page - 页码
 * @param {number} data.pageSize - 每页数量
 */
export async function getDataFileList(data) {
    return post('/api/data-file/list', data, requestInterceptor());
}

/**
 * 获取目录树
 */
export async function getDirectoryTree() {
    return get('/api/data-file/tree', requestInterceptor());
}

/**
 * 创建目录
 * @param {Object} data - 目录信息
 * @param {string} data.path - 父目录路径
 * @param {string} data.name - 目录名称
 */
export async function createDirectory(data) {
    return post('/api/data-file/create-dir', data, requestInterceptor());
}

/**
 * 删除数据文件或目录
 * @param {Object} data - 删除参数
 * @param {string} data.path - 文件/目录路径
 * @param {string} data.name - 文件/目录名称
 * @param {boolean} data.isDir - 是否为目录
 */
export async function deleteDataFile(data) {
    return post('/api/data-file/delete', data, requestInterceptor());
}

/**
 * 上传数据文件（支持指定目录）
 * @param {FormData} formData - 文件表单数据
 */
export async function uploadDataFile(formData) {
    return post('/api/data-file/upload', formData, 
        requestInterceptor(),
        'form-data'
    );
}

/**
 * 下载数据文件（使用a标签方式，浏览器管理下载）
 * @param {string} fileName - 文件名
 * @param {string} filePath - 文件路径
 */
export async function downloadDataFile(fileName, filePath) {
    try {
        // 1. 获取临时下载链接
        const res = await post('/api/data-file/download', {
            filename: fileName,
            path: filePath
        }, requestInterceptor())
        
        if (res.code === 0) {
            const { downloadUrl, fileName } = res.data
            
            // 2. 创建隐藏的a标签进行下载
            const link = document.createElement('a')
            link.href = downloadUrl
            link.download = fileName
            link.style.display = 'none'
            
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            
            return true
        } else {
            throw new Error(res.msg || '获取下载链接失败')
        }
    } catch (error) {
        console.error('下载文件失败:', error)
        throw error
    }
}

/**
 * 批量下载文件（支持进度显示）
 * @param {Array} files - 文件列表
 */
export async function batchDownloadFiles(files) {
    const downloadPromises = files.map(file => 
        downloadDataFile(file.fileName, file.path)
    )
    
    try {
        await Promise.all(downloadPromises)
        return true
    } catch (error) {
        console.error('批量下载失败:', error)
        throw error
    }
}

/**
 * 批量删除数据文件
 * @param {Array} files - 文件列表，包含 fileName, path, isDir
 */
export async function batchDeleteDataFile(files) {
    return post('/api/data-file/batch-delete', { files }, requestInterceptor());
}
