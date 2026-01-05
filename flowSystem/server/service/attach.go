// 文件上传
package service

import (
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type UploadOption struct {
	ID       string // 上传文件的用户ID, 可有可无, string更方便
	FileType int    // 文件类型，1：图片，2：压缩包
	MaxSize  int64  // 最大文件大小，单位：字节 byte
}

// 文件上传
func UploadFile(fileHeader *multipart.FileHeader, dstPath string, opt *UploadOption) (string, error) {
	var id string

	if fileHeader == nil { // 检查文件是否为空
		return "", errors.New("文件为空")
	}

	// 文件类型
	suffix := filepath.Ext(fileHeader.Filename)
	fileType := fileHeader.Header.Get("Content-Type")

	if opt != nil {
		id = opt.ID

		switch opt.FileType { // 检查文件类型
		case 1: // 图片
			if suffix != ".jpg" && suffix != ".jpeg" && suffix != ".png" {
				return "", errors.New("文件类型必须为图片")
			}
			if fileType != "image/jpeg" && fileType != "image/png" {
				return "", errors.New("文件类型必须为图片")
			}
		case 2: // 压缩包
		default:
		}

		// 检查文件大小是否超过限制，单位：字节 byte
		if opt.MaxSize != 0 && fileHeader.Size > opt.MaxSize {
			return "", errors.New("文件大小不能超过" + strconv.FormatInt(opt.MaxSize, 10))
		}
	}

	// 确保上传目录存在
	if err := os.MkdirAll(dstPath, 0755); err != nil {
		return "", fmt.Errorf("创建上传目录失败: %v", err)
	}

	// 打开文件
	file, err := fileHeader.Open()
	if err != nil {
		return "", errors.New("打开文件失败")
	}
	defer file.Close()

	// 创建目标文件
	var format string
	if id == "" {
		format = "%s%s%s"
	} else {
		format = "%s-%s%s"
	}
	filename := fmt.Sprintf(format, time.Now().Format("20060102150405"), id, suffix)
	path := filepath.Join(dstPath, filename)
	dstFile, err := os.Create(path)
	if err != nil {
		return "", errors.New("创建文件失败")
	}
	defer dstFile.Close()

	// 复制文件内容到目标文件
	io.Copy(dstFile, file)

	return path, nil // 返回文件路径
}
