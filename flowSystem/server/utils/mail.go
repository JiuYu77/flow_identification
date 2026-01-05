package utils

import (
	"errors"
	"fmt"

	"flow-server/config"

	goemail "github.com/JiuYu77/go-email"
)

var cfg goemail.Config
var Verifier *goemail.Verifier

func InitVerfier() {
	cfg = goemail.Config{
		SMTPConfig: goemail.SMTPConfig{
			Host:     config.SysConfig.SMTP.Host,
			Port:     config.SysConfig.SMTP.Port, // 25 465 587
			Username: config.SysConfig.SMTP.Username,
			Password: config.SysConfig.SMTP.Password,
			From:     config.SysConfig.SMTP.From,
		},
		CodeExpiry:   config.SysConfig.Verifier.Expiry,
		CacheCleanup: config.SysConfig.Verifier.Cleanup,
	}
	Verifier = goemail.NewVerifier(&cfg)
}

func BuildVerificationCode(to string, code string) ([]byte, error) {
	if to == "" {
		return nil, errors.New("收件人不能为空")
	}

	from := cfg.From
	subject := "验证码"

	body := fmt.Sprintf("您的验证码为: %s, 有效期为 %v 分钟", code, cfg.CodeExpiry.Minutes())
	msg := fmt.Sprintf("Subject: %s\r\nFrom: %s\r\nTo: %s\r\n\r\n%s", subject, from, to, body)

	return []byte(msg), nil
}
