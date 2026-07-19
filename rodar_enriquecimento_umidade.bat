@echo off
REM Enriquecimento NASA POWER (umidade) — retoma automaticamente do cache + CSV parcial.
REM Este arquivo inicia Python numa NOVA janela minimizada; fechar o Cursor NAO encerra esse processo.
REM Logs: logs\umidade_enriquecimento*.log

cd /d "%~dp0"
if not exist "logs" mkdir logs

echo [%date% %time%] === Inicio sessao enriquecimento ===>> "logs\umidade_enriquecimento_err.log"

REM /MIN = janela minimizada | titulo permite identificar no Gerenciador de Tarefas
start "NASA POWER - Umidade TCC-2" /MIN cmd /c ^
  "python scripts\enriquecer_dados_umidade.py --input base_de_dados_com_historico.csv --output base_de_dados_com_umidade.csv --max_workers 4 --salvar_cache_periodicamente 100 >> logs\umidade_enriquecimento.log 2>> logs\umidade_enriquecimento_err.log"

echo.
echo Processo iniciado em janela separada (minimizada).
echo Acompanhe: logs\umidade_enriquecimento_err.log
echo Para encerrar: Gerenciador de Tarefas ^> procurar "python.exe" ou titulo da janela.
echo AVISO: nao rode duas instancias ao mesmo tempo (risco de sobrescrever cache/CSV).
echo.
pause
