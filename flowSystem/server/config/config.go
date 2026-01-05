package config

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/goccy/go-yaml"
)

type MySQLConfig struct {
	User      string `yaml:"user"`
	Password  string `yaml:"password"`
	Host      string `yaml:"host"`
	Port      int    `yaml:"port"`
	Database  string `yaml:"database"`
	Charset   string `yaml:"charset"`
	ParseTime string `yaml:"parseTime"`
	Loc       string `yaml:"loc"`
	DSN       string
}

type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
	Addr string
}
type RedisConfig struct {
	Host         string `yaml:"host"`
	Port         int    `yaml:"port"`
	Password     string `yaml:"password"`
	DB           int    `yaml:"db"`
	PoolSize     int    `yaml:"poolSize"`
	MinIdleConns int    `yaml:"minIdleConns"`
	Addr         string
}

type SMTPConfig struct {
	Host     string `yaml:"host"`     // SMTP 服务器主机名
	Port     int    `yaml:"port"`     // SMTP 服务器端口号
	Username string `yaml:"username"` // SMTP 服务器用户名
	Password string `yaml:"password"` // SMTP 服务器密码
	From     string `yaml:"from"`     // 发件人邮箱地址
}

type VerifierConfig struct {
	Expiry  time.Duration `yaml:"expiry"`  // 验证码 过期时间
	Cleanup time.Duration `yaml:"cleanup"` // 缓存清理时间间隔
}

type AuthConfig struct {
	JwtKey             string        `yaml:"jwtKey"`             // JWT 密钥
	AccessTokenExpire  time.Duration `yaml:"accessTokenExpire"`  // 访问令牌过期时间
	RefreshTokenExpire time.Duration `yaml:"refreshTokenExpire"` // 刷新令牌过期时间
}

type DataConfig struct {
	RootDir string `yaml:"rootDir"` // 数据根目录
	JwtKey  string `yaml:"jwtKey"`  // JWT 密钥
}

type Config struct {
	MySQL    MySQLConfig
	Server   ServerConfig
	Redis    RedisConfig
	SMTP     SMTPConfig
	Verifier VerifierConfig
	Auth     AuthConfig `yaml:"auth"` // 认证配置
	Data     DataConfig // 数据配置
}

var SysConfig Config

func InitConfig() {
	byteData, err := os.ReadFile("config/flow_system.yaml")
	if err != nil {
		panic("fail to read config file: " + err.Error())
	}

	err = yaml.Unmarshal(byteData, &SysConfig)
	if err != nil {
		panic("fail to unmarshal config file: " + err.Error())
	}

	// mysql dsn
	mysqlConfig := SysConfig.MySQL
	SysConfig.MySQL.DSN = fmt.Sprintf("%s:%s@tcp(%s:%d)/%s?charset=%s&parseTime=%s&loc=%s",
		mysqlConfig.User, mysqlConfig.Password, mysqlConfig.Host, mysqlConfig.Port, mysqlConfig.Database,
		mysqlConfig.Charset, mysqlConfig.ParseTime, mysqlConfig.Loc)

	// server addr
	SysConfig.Server.Addr = fmt.Sprintf("%s:%d", SysConfig.Server.Host, SysConfig.Server.Port)

	// redis addr
	SysConfig.Redis.Addr = fmt.Sprintf("%s:%d", SysConfig.Redis.Host, SysConfig.Redis.Port)

	log.Println("加载配置文件成功.")
	fmt.Println(SysConfig)
}
