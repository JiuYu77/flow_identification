package jwt

import (
	"errors"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

var (
	JWTdm *JWTDowloadManager
)

type JWTDowloadManager struct {
	secretKey []byte // JWT密钥
}

func NewJWTDowloadManager(secret string) *JWTDowloadManager {
	return &JWTDowloadManager{
		secretKey: []byte(secret),
	}
}

// 生成下载令牌
func (m *JWTDowloadManager) GenerateDownloadToken(fileName, filePath string, expires time.Duration) (string, error) {
	claims := jwt.MapClaims{
		"fileName": fileName,
		"filePath": filePath,
		"exp":      jwt.NewNumericDate(time.Now().Add(expires)), // 过期时间
	}
	jwt := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	token, err := jwt.SignedString(m.secretKey)
	if err != nil {
		return "", err
	}
	return token, nil
}

// 验证下载令牌
func (m *JWTDowloadManager) ParseDownloadToken(tokenStr string) (jwt.MapClaims, error) {
	claims := jwt.MapClaims{}
	token, err := jwt.ParseWithClaims(tokenStr, &claims, func(token *jwt.Token) (any, error) {
		return m.secretKey, nil
	})
	if err != nil {
		return nil, err
	}
	if !token.Valid {
		return nil, errors.New("invalid token")
	}
	return claims, nil
}
