package jwt

import "flow-server/config"

func InitJWT() {
	jwtKey = []byte(config.SysConfig.Auth.JwtKey)
	AccessTokenExpire = config.SysConfig.Auth.AccessTokenExpire
	RefreshTokenExpire = config.SysConfig.Auth.RefreshTokenExpire

	JWTdm = NewJWTDowloadManager(config.SysConfig.Data.JwtKey)
}
