package service

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// RemoveFile 删除单个文件
func RemoveFile(filePath string) error {
	if filePath == "" {
		return errors.New("文件路径为空，无需删除")
	}

	absPath, err := filepath.Abs(filePath) // （可选）转为绝对路径，避免相对路径歧义
	if err != nil {
		return fmt.Errorf("获取绝对路径失败：%w", err)
	}

	// 安全检查：避免删除系统关键目录
	if isSystemPath(absPath) {
		return errors.New("禁止删除系统关键路径")
	}

	err = os.Remove(absPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("文件不存在：%w", err)
		}
		return fmt.Errorf("删除文件失败：%w", err)
	}

	return nil
}

// RemoveFiles 批量删除文件
func RemoveFiles(filePaths []string) error {
	for _, filePath := range filePaths {
		if err := RemoveFile(filePath); err != nil {
			return fmt.Errorf("删除文件 %s 失败: %w", filePath, err)
		}
	}
	return nil
}

// RemoveDir 删除目录（递归删除）
func RemoveDir(dirPath string) error {
	if dirPath == "" {
		return errors.New("目录路径为空")
	}

	absPath, err := filepath.Abs(dirPath)
	if err != nil {
		return fmt.Errorf("获取绝对路径失败：%w", err)
	}

	// 安全检查
	if isSystemPath(absPath) {
		return errors.New("禁止删除系统关键目录")
	}

	return os.RemoveAll(absPath)
}

// isSystemPath 安全检查函数
func isSystemPath(path string) bool {
	systemPaths := []string{
		"/", "/bin", "/usr", "/etc", "/var", "/lib", "/opt",
		"/home", "/root", "/tmp", "/proc", "/sys", "/dev",
	}

	for _, sysPath := range systemPaths {
		if path == sysPath || len(path) > len(sysPath) && path[:len(sysPath)] == sysPath {
			return true
		}
	}
	return false
}
