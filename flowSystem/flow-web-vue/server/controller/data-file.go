package controller

import (
	"flow-server/config"
	"flow-server/service"

	"github.com/gin-gonic/gin"
)

func InitDataFile() {
	service.RootDir = config.SysConfig.Data.RootDir
}

// 获取数据文件列表
func GetDataFileList(c *gin.Context) {
	service.GetDataFileList(c)
}

func GetDataFileTree(c *gin.Context) {
	service.GetDirectoryTree(c)
}

func CreateDataDir(c *gin.Context) {
	service.CreateDirectory(c)
}

func DeleteDataFile(c *gin.Context) {
	service.DeleteDataFile(c)
}

func UploadDataFile(c *gin.Context) {
	service.UploadDataFile(c)
}
func DownloadDataFile(c *gin.Context) {
	service.DownloadDataFile(c)
}
func DownloadFileByToken(c *gin.Context) {
	service.DownloadFileByToken(c)
}

func BatchDeleteDataFile(c *gin.Context) {
	service.BatchDeleteDataFile(c)
}
