package service

import (
	"errors"
	"flow-server/jwt"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"golang.org/x/crypto/bcrypt"
)

// HashPassword 加密密码
func HashPassword(password string) (string, error) {
	// bcrypt.DefaultCost 是推荐的加密成本(10)
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", err
	}
	return string(hashedPassword), nil
}

// CheckPassword 验证密码
func CheckPassword(password, hashedPassword string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hashedPassword), []byte(password))
	return err == nil
}

func IsValidPassword(password string) error {
	// 检查密码是否为空
	if password == "" {
		return errors.New("密码不能为空")
	}
	// 检查密码长度是否符合要求（例如：8-25个字符）
	if len(password) < 8 || len(password) > 25 {
		return errors.New("密码长度必须在8-25个字符之间")
	}
	// 检查密码是否包含不允许的特殊字符
	allowedChars := regexp.MustCompile(`^[a-zA-Z0-9@#$%^&*()_+\-=\[\]{};:,.<>?]+$`) // 定义允许的字符集（字母、数字、部分特殊字符）
	if !allowedChars.MatchString(password) {
		return errors.New("密码包含不允许的特殊字符")
	}
	return nil
}

func setRefreshTokenCookie(c *gin.Context, refreshToken string) {
	c.SetCookie("refresh_token", refreshToken, int(jwt.RefreshTokenExpire.Seconds()), "/", "", false, true)
}

func getRefreshTokenFromCookie(c *gin.Context) (string, error) {
	refreshToken, err := c.Cookie("refresh_token")
	if err != nil {
		return "", err
	}
	return refreshToken, nil
}

func getUserIDFromHeader(c *gin.Context) (uint, error) {
	idStr := c.GetHeader("X-User-ID")
	id, err := strconv.ParseUint(idStr, 10, 32) // idStr为空会返回错误
	if err != nil {
		return 0, fmt.Errorf("获取用户信息失败")
	}
	return uint(id), nil
}

func getAccessTokenFromHeader(c *gin.Context) (string, error) {
	authHeader := c.GetHeader("Authorization")
	if authHeader == "" {
		return "", fmt.Errorf("获取访问令牌失败")
	}

	parts := strings.SplitN(authHeader, " ", 2)
	var accessToken string
	if len(parts) == 2 && strings.ToLower(parts[0]) == "bearer" {
		accessToken = parts[1] // 支持 "Bearer <token>" 格式
	} else {
		accessToken = authHeader
	}
	return accessToken, nil
}

// 检查用户名是否合法
func IsValidUsername(username string) error {
	if username == "" {
		return errors.New("用户名不能为空")
	}
	if len(username) < 3 || len(username) > 20 {
		return errors.New("用户名长度必须在3-20个字符之间")
	}
	allowedChars := regexp.MustCompile(`^[a-zA-Z0-9_-]+$`) // 定义允许的字符集（字母、数字、下划线、中划线）
	if !allowedChars.MatchString(username) {
		return errors.New("用户名包含不允许的特殊字符")
	}
	return nil
}

func CheckAccessToken(c *gin.Context) (*jwt.Claims, error) {
	accessToken, err := getAccessTokenFromHeader(c)
	if err != nil {
		return nil, err
	}
	claims, err := jwt.CheckToken(accessToken, "access")
	if err != nil {
		return nil, err
	}
	return claims, nil
}

func CheckRefreshToken(c *gin.Context) (*jwt.Claims, error) {
	refreshToken, err := getRefreshTokenFromCookie(c) // 从请求中获取刷新令牌
	if err != nil {
		return nil, errors.New("未获取到刷新令牌")
	}

	claims, err := jwt.CheckToken(refreshToken, "refresh")
	if err != nil {
		return nil, err
	}
	return claims, nil
}

func MaskEmail(email string) string {
	// 检查邮箱是否为空
	if email == "" {
		return "***"
	}
	// 掩码用户名
	username, domain := strings.Split(email, "@")[0], strings.Split(email, "@")[1]
	if len(username) > 7 {
		username = username[:4] + strings.Repeat("*", len(username)-4)
	} else if len(username) > 5 {
		username = username[:3] + strings.Repeat("*", len(username)-3)
	} else if len(username) > 2 {
		username = username[:2] + strings.Repeat("*", len(username)-2)
	} else {
		username = strings.Repeat("*", len(username))
	}
	return fmt.Sprintf("%s@%s", username, domain)
}
