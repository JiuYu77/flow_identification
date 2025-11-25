// 时间格式化函数
export const formatTime = (timeString) => {
    if (!timeString) return ''

    try{
        const date = new Date(timeString);
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        }).replace(/\//g, '-');
    } catch(error){
        return timeString;
    }
}
