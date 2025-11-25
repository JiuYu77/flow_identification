package service

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"flow-server/jwt"
	"flow-server/resp"

	"github.com/gin-gonic/gin"
)

type DataFileListRequest struct {
	Keyword  string `json:"keyword"`
	FileType string `json:"fileType"`
	Page     int    `json:"page"`
	PageSize int    `json:"pageSize"`
	Path     string `json:"path"` // 新增：目录路径
}

type DataFile struct {
	ID         uint       `json:"id"`
	FileName   string     `json:"fileName"`
	FileType   string     `json:"fileType"`
	FileSize   int64      `json:"fileSize"`
	UploadTime time.Time  `json:"uploadTime"`
	Status     int        `json:"status"`
	IsDir      bool       `json:"isDir"`              // 新增：是否为目录
	Path       string     `json:"path"`               // 新增：文件路径
	Children   []DataFile `json:"children,omitempty"` // 新增：子文件/目录
}

var RootDir string

// 获取数据文件列表（支持层级目录）
func GetDataFileList(c *gin.Context) {
	var req DataFileListRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		resp.FailWithMsg(c, "参数错误")
		return
	}

	// 基础目录
	baseDir := RootDir
	if req.Path != "" {
		baseDir = filepath.Join(baseDir, req.Path)
	}

	// 检查目录是否存在
	if _, err := os.Stat(baseDir); os.IsNotExist(err) {
		resp.OKWithData(c, gin.H{
			"fileList": []DataFile{},
			"total":    0,
		})
		return
	}

	// 读取目录内容
	entries, err := os.ReadDir(baseDir)
	if err != nil {
		resp.FailWithMsg(c, "读取目录失败")
		return
	}

	var fileList []DataFile
	for _, entry := range entries {
		info, err := entry.Info()
		if err != nil {
			continue
		}

		// 过滤隐藏文件
		if strings.HasPrefix(entry.Name(), ".") {
			continue
		}

		file := DataFile{
			FileName:   entry.Name(),
			IsDir:      entry.IsDir(),
			Path:       req.Path,
			UploadTime: info.ModTime(),
		}

		if !entry.IsDir() {
			file.FileSize = info.Size()
			file.FileType = strings.ToLower(strings.TrimPrefix(filepath.Ext(entry.Name()), "."))
			if file.FileType == "" {
				file.FileType = "unknown"
			}
			file.Status = 0
		}

		// 应用搜索过滤
		if req.Keyword != "" && !strings.Contains(strings.ToLower(entry.Name()), strings.ToLower(req.Keyword)) {
			continue
		}
		if req.FileType != "" && !entry.IsDir() && file.FileType != req.FileType {
			continue
		}

		fileList = append(fileList, file)
	}

	resp.OKWithData(c, gin.H{
		"fileList":    fileList,
		"total":       len(fileList),
		"currentPath": req.Path,
	})
}

// 创建目录
func CreateDirectory(c *gin.Context) {
	var req struct {
		Path string `json:"path"`
		Name string `json:"name"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		resp.FailWithMsg(c, "参数错误")
		return
	}

	baseDir := RootDir
	fullPath := filepath.Join(baseDir, req.Path, req.Name)

	if err := os.MkdirAll(fullPath, 0755); err != nil {
		resp.FailWithMsg(c, "创建目录失败")
		return
	}

	resp.OKWithMsg(c, "目录创建成功")
}

// 获取目录树（用于前端树形组件）
func GetDirectoryTree(c *gin.Context) {
	baseDir := RootDir

	tree, err := buildDirectoryTree(baseDir, "")
	if err != nil {
		resp.FailWithMsg(c, "获取目录树失败")
		return
	}

	resp.OKWithData(c, gin.H{
		"tree": tree,
	})
}

// 构建目录树
func buildDirectoryTree(baseDir, relativePath string) ([]DataFile, error) {
	fullPath := filepath.Join(baseDir, relativePath)

	entries, err := os.ReadDir(fullPath)
	if err != nil {
		return nil, err
	}

	var tree []DataFile
	for _, entry := range entries {
		if !entry.IsDir() || strings.HasPrefix(entry.Name(), ".") {
			continue
		}

		childPath := filepath.Join(relativePath, entry.Name())
		info, err := entry.Info()
		if err != nil {
			continue
		}

		node := DataFile{
			FileName:   entry.Name(),
			IsDir:      true,
			Path:       relativePath,
			UploadTime: info.ModTime(),
		}

		// 递归获取子目录
		children, err := buildDirectoryTree(baseDir, childPath)
		if err == nil && len(children) > 0 {
			node.Children = children
		}

		tree = append(tree, node)
	}

	return tree, nil
}

// 上传数据文件（支持指定目录）
func UploadDataFile(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		resp.FailWithMsg(c, "文件上传失败")
		return
	}

	// 获取上传目录参数
	uploadPath := c.PostForm("path")
	if uploadPath == "" {
		uploadPath = "."
	}

	// 构建上传目录路径
	uploadDir := filepath.Join(RootDir, uploadPath)

	// 上传文件
	filePath, err := UploadFile(file, uploadDir, &UploadOption{MaxSize: 1024 * 1024 * 10}) // 文件大小不超过10MB
	if err != nil {
		resp.FailWithMsg(c, "文件上传失败")
		return
	}

	resp.OKWithData(c, gin.H{
		"fileName": file.Filename,
		"filePath": filePath,
		"fileSize": file.Size,
		"path":     uploadPath,
	})
}

type DownloadDataFileRequest struct {
	FileName string `json:"filename"`
	Path     string `json:"path"`
}

// 下载数据文件
func DownloadDataFile(c *gin.Context) {
	var req DownloadDataFileRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		resp.FailWithMsg(c, "参数错误")
		return
	}

	// 构建文件路径
	filePath := filepath.Join(RootDir, req.Path, req.FileName)

	// 检查文件是否存在
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		resp.FailWithMsg(c, "文件不存在")
		return
	}

	// 生成下载令牌
	token, err := jwt.JWTdm.GenerateDownloadToken(req.FileName, filePath, time.Hour) // 令牌有效期1小时
	if err != nil {
		resp.FailWithMsg(c, "生成下载令牌失败")
		return
	}

	resp.OKWithData(c, gin.H{
		"downloadUrl": fmt.Sprintf("/api/data-file/download/%s", token), // 临时下载链接, 下载URL包含令牌
		"fileName":    req.FileName,
	})
}

func DownloadFileByToken(c *gin.Context) {
	token := c.Param("token")
	if token == "" {
		resp.FailWithMsg(c, "令牌不能为空")
		return
	}

	// 验证令牌
	claims, err := jwt.JWTdm.ParseDownloadToken(token)
	if err != nil {
		resp.FailWithMsg(c, "令牌无效")
		return
	}

	filePath := claims["filePath"].(string)
	fileName := claims["fileName"].(string)

	// 检查文件是否存在
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		resp.FailWithMsg(c, "文件不存在")
		return
	}

	// 下载文件
	c.FileAttachment(filePath, fileName)
}

// 删除数据文件或目录
func DeleteDataFile(c *gin.Context) {
	var req struct {
		ID    uint   `json:"id"`
		Path  string `json:"path"`
		Name  string `json:"name"`
		IsDir bool   `json:"isDir"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		resp.FailWithMsg(c, "参数错误")
		return
	}

	baseDir := RootDir
	fullPath := filepath.Join(baseDir, req.Path, req.Name)

	if req.IsDir {
		// 删除目录
		if err := os.RemoveAll(fullPath); err != nil {
			resp.FailWithMsg(c, "删除目录失败")
			return
		}
	} else {
		// 删除文件
		if err := os.Remove(fullPath); err != nil {
			resp.FailWithMsg(c, "删除文件失败")
			return
		}
	}

	resp.OKWithMsg(c, "删除成功")
}

// 批量删除数据文件或目录
func BatchDeleteDataFile(c *gin.Context) {
	var req struct {
		Files []struct {
			FileName string `json:"fileName"`
			Path     string `json:"path"`
			IsDir    bool   `json:"isDir"`
		} `json:"files"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		resp.FailWithMsg(c, "参数错误")
		return
	}

	if len(req.Files) == 0 {
		resp.FailWithMsg(c, "请选择要删除的文件")
		return
	}

	baseDir := RootDir
	successCount := 0
	failedFiles := []string{}

	for _, file := range req.Files {
		fullPath := filepath.Join(baseDir, file.Path, file.FileName)

		var err error
		if file.IsDir {
			// 删除目录
			err = os.RemoveAll(fullPath)
		} else {
			// 删除文件
			err = os.Remove(fullPath)
		}

		if err != nil {
			failedFiles = append(failedFiles, file.FileName)
			fmt.Printf("删除失败 %s: %v\n", fullPath, err)
		} else {
			successCount++
		}
	}

	if len(failedFiles) > 0 {
		resp.OKWithData(c, gin.H{
			"successCount": successCount,
			"failedCount":  len(failedFiles),
			"failedFiles":  failedFiles,
			"message":      fmt.Sprintf("成功删除 %d 个，失败 %d 个", successCount, len(failedFiles)),
		})
	} else {
		resp.OKWithMsg(c, fmt.Sprintf("成功删除 %d 个文件/文件夹", successCount))
	}
}
