package http_server

import (
	"log"
	"net/http"
)

const (
	// 字体颜色 basic color
	ColorBlack   = "\033[30m"
	ColorRed     = "\033[31m"
	ColorGreen   = "\033[32m"
	ColorYellow  = "\033[33m"
	ColorBlue    = "\033[34m"
	ColorMagenta = "\033[35m"
	ColorCyan    = "\033[36m"
	ColorWhite   = "\033[37m"

	// 字体颜色 亮色 bright color
	BrightColorBlack   = "\033[90m"
	BrightColorRed     = "\033[91m"
	BrightColorGreen   = "\033[92m"
	BrightColorYellow  = "\033[93m"
	BrightColorBlue    = "\033[94m"
	BrightColorMagenta = "\033[95m"
	BrightColorCyan    = "\033[96m"
	BrightColorWhite   = "\033[97m"

	// 背景颜色 background color
	BgColorBlack   = "\033[40m"
	BgColorRed     = "\033[41m"
	BgColorGreen   = "\033[42m"
	BgColorYellow  = "\033[43m"
	BgColorBlue    = "\033[44m"
	BgColorMagenta = "\033[45m"
	BgColorCyan    = "\033[46m"
	BgColorWhite   = "\033[47m"

	// 字体样式 font style 、misc
	FontBold          = "\033[1m"
	FontHalfBright    = "\033[2m" // 低亮
	FontItalic        = "\033[3m"
	FontUnderline     = "\033[4m"
	FontBlink         = "\033[5m" // 闪烁（无效）
	FontReverse       = "\033[7m" // 反显
	FontBlanking      = "\033[8m" // 消隐
	FontStrikeThrough = "\033[9m" // 删除线

	// 重置
	Reset = "\033[0m"
)

const (
	LogFormatFound            = ColorGreen + "%-20s" + Reset
	FoundMsg                  = "Found"
	LogFormatMethodNotAllowed = ColorRed + "%-20s" + Reset
	MethodNotAllowedMsg       = "Method Not Allowed"
	LogFormatNotFound         = ColorRed + "%-20s" + Reset
	NotFoundMsg               = "Not Found"

	LogGET  = BgColorBlue + BrightColorWhite + "%-6s" + Reset
	LogPOST = BgColorCyan + BrightColorWhite + "%-6s" + Reset
	LogPath = "\"%s\""

	Log200 = BgColorGreen + BrightColorWhite + " 200 " + Reset
	Log404 = BgColorRed + BrightColorWhite + " 404 " + Reset
	Log405 = BgColorYellow + BrightColorWhite + " 405 " + Reset
)

func LogFound(method, path string) {
	switch method {
	case http.MethodGet:
		LogFoundGET(method, path)
	case http.MethodPost:
		LogFoundPOST(method, path)
	default:
		log.Printf(LogFormatFound+" "+LogGET+" "+LogPath, FoundMsg, method, path)
	}
}

func LogNotFound(method, path string) {
	switch method {
	case http.MethodGet:
		LogNotFoundGET(method, path)
	default:
	}
}

func LogNotAllowed(method, path string) {
	switch method {
	case http.MethodGet:
		LogMethodNotAllowed(method, path)
	default:
	}
}

func LogFoundGET(method, path string) {
	log.Printf(Log200+" "+LogFormatFound+" "+LogGET+" "+LogPath, FoundMsg, method, path)
}
func LogFoundPOST(method, path string) {
	log.Printf(Log200+" "+LogFormatFound+" "+LogPOST+" "+LogPath, FoundMsg, method, path)
}

func LogMethodNotAllowed(method, path string) {
	log.Printf(Log405+" "+LogFormatMethodNotAllowed+" "+LogGET+" "+LogPath, MethodNotAllowedMsg, method, path)
}

func LogNotFoundGET(method, path string) {
	log.Printf(Log404+" "+LogFormatNotFound+" "+LogGET+" "+LogPath, NotFoundMsg, method, path)
}

func LogRun(addr string) {
	log.Println("[HTTP] Listening on", addr) // log
	log.Println("Press Ctrl+C to stop")      // log
}
func LogRunTLS(addr string) {
	log.Println("[HTTPS] Listening on", addr) // log
	log.Println("Press Ctrl+C to stop")       // log
}
