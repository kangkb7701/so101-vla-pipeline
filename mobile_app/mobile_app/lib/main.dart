import 'dart:async';
import 'dart:convert';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_database/firebase_database.dart';
import 'package:flutter/material.dart';
import 'package:flutter_mjpeg/flutter_mjpeg.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

import 'firebase_options.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
  } catch (e) {
    debugPrint('Firebase init skipped: $e');
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Robot Control',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.light,
        scaffoldBackgroundColor: const Color(0xFFF8FAFC),
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF16A34A)),
        useMaterial3: true,
      ),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final TextEditingController _ipController = TextEditingController(text: '192.168.0.26');
  final TextEditingController _portController = TextEditingController(text: '8000');
  final TextEditingController _commandController = TextEditingController();

  bool _isStreaming = true;
  bool _isListening = false;
  bool _robotRunning = false;
  bool _taskSucceeded = false;
  String? _lastSuccessTs;
  String _selectedBasket = 'green';
  String _selectedView = 'split';
  String _text = '음성 명령 대기 중...';
  String _speechStatus = '대기 중';
  String _serverStatus = '';
  String _latestInstruction = '(없음)';
  String _latestCommand = '(없음)';
  int _selectedTab = 0;
  bool _historyLoading = false;
  List<Map<String, dynamic>> _commandHistory = [];
  final List<String> _statusHistory = ['서버 연결 대기'];

  DatabaseReference? _dbRef;
  final stt.SpeechToText _speech = stt.SpeechToText();
  Timer? _latestTimer;

  @override
  void initState() {
    super.initState();
    _requestPermissions();
    _selectBasket('green', updateOnly: true);
    _checkServerHealth();
    _fetchLatest();
    _latestTimer = Timer.periodic(const Duration(seconds: 2), (_) => _fetchLatest());
    try {
      _dbRef = FirebaseDatabase.instance.ref();
    } catch (e) {
      debugPrint('Firebase ref unavailable: $e');
    }
  }

  @override
  void dispose() {
    _latestTimer?.cancel();
    _ipController.dispose();
    _portController.dispose();
    _commandController.dispose();
    super.dispose();
  }

  Future<void> _requestPermissions() async {
    await [Permission.microphone, Permission.camera].request();
  }

  String get _baseUrl => 'http://${_ipController.text.trim()}:${_portController.text.trim()}';

  String _basketCommand(String color) => 'pick the banana and place it in the $color basket';

  void _pushStatus(String text) {
    setState(() {
      _statusHistory.insert(0, text);
      if (_statusHistory.length > 5) _statusHistory.removeLast();
    });
  }

  void _selectBasket(String color, {bool updateOnly = false}) {
    setState(() {
      _selectedBasket = color;
      _commandController.text = _basketCommand(color);
    });
    if (!updateOnly) _pushStatus('$color basket 선택');
  }

  void _toggleStream() {
    setState(() {
      _isStreaming = !_isStreaming;
      _serverStatus = _isStreaming ? '스트리밍 시작' : '스트리밍 중단';
    });
    if (_isStreaming) {
      _checkServerHealth();
      _fetchLatest();
      _latestTimer?.cancel();
      _latestTimer = Timer.periodic(const Duration(seconds: 2), (_) => _fetchLatest());
    } else {
      _latestTimer?.cancel();
      _latestTimer = null;
      _pushStatus('카메라 스트리밍 중단');
    }
  }

  Future<void> _fetchLatest() async {
    try {
      final res = await http.get(Uri.parse('$_baseUrl/command/latest')).timeout(const Duration(seconds: 3));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        final instr = data['instruction'] as Map<String, dynamic>?;
        final last = data['last_command'];
        final success = data['success'] as Map<String, dynamic>?;
        final successTs = success?['ts']?.toString();
        final isNewSuccess = successTs != null && successTs.isNotEmpty && successTs != _lastSuccessTs;
        setState(() {
          _latestInstruction = instr == null ? '(없음)' : '${instr['text'] ?? ''}  @${instr['ts'] ?? '-'}';
          _latestCommand = last == null ? '(없음)' : last.toString();
          if (isNewSuccess) {
            _lastSuccessTs = successTs;
            _robotRunning = false;
            _taskSucceeded = true;
            _serverStatus = '작업 성공';
          }
        });
        if (isNewSuccess) _pushStatus('작업 성공: ${success?['task'] ?? ''}');
      }
    } catch (e) {
      debugPrint('fetchLatest fail: $e');
    }
  }

  Future<void> _fetchHistory() async {
    setState(() => _historyLoading = true);
    try {
      final res = await http.get(Uri.parse('$_baseUrl/command/history?limit=50')).timeout(const Duration(seconds: 3));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        final items = (data['items'] as List? ?? [])
            .whereType<Map>()
            .map((item) => Map<String, dynamic>.from(item))
            .toList()
            .reversed
            .toList();
        setState(() {
          _commandHistory = items;
          _serverStatus = '제어 기록 ${items.length}개 불러옴';
        });
      } else {
        setState(() => _serverStatus = '제어 기록 응답 ${res.statusCode}');
      }
    } catch (e) {
      setState(() => _serverStatus = '제어 기록 불러오기 실패: $e');
    } finally {
      if (mounted) setState(() => _historyLoading = false);
    }
  }

  void _selectTab(int index) {
    setState(() => _selectedTab = index);
    if (index == 1) _fetchHistory();
  }

  Future<void> _checkServerHealth() async {
    try {
      final res = await http.get(Uri.parse('$_baseUrl/health')).timeout(const Duration(seconds: 3));
      if (res.statusCode == 200) {
        setState(() => _serverStatus = '서버 연결됨');
        _pushStatus('서버 연결됨');
      } else {
        setState(() => _serverStatus = '서버 응답 ${res.statusCode}');
        _pushStatus('서버 응답 ${res.statusCode}');
      }
    } catch (e) {
      setState(() => _serverStatus = '서버 연결 실패: $e');
      _pushStatus('서버 연결 실패');
    }
  }

  void _listen() async {
    if (!_isListening) {
      final available = await _speech.initialize(
        onStatus: (val) {
          debugPrint('onStatus: $val');
          if (!mounted) return;
          setState(() {
            _speechStatus = val == 'done' || val == 'notListening' ? '음성 입력 종료' : '듣는 중';
            if (val == 'done' || val == 'notListening') _isListening = false;
          });
        },
        onError: (val) {
          debugPrint('onError: $val');
          if (!mounted) return;
          setState(() {
            _isListening = false;
            _speechStatus = '음성 인식 오류';
          });
          _pushStatus('음성 인식 오류');
        },
      );
      if (available) {
        setState(() {
          _isListening = true;
          _speechStatus = '듣는 중';
          _text = '말해주세요...';
        });
        _pushStatus('음성 입력 시작');
        _speech.listen(
          localeId: 'en_US',
          listenFor: const Duration(seconds: 60),
          pauseFor: const Duration(seconds: 15),
          onResult: (val) {
            final words = val.recognizedWords.trim();
            if (words.isEmpty) return;
            setState(() {
              _text = words;
              _commandController.text = words;
              _commandController.selection = TextSelection.collapsed(offset: _commandController.text.length);
              _speechStatus = val.finalResult ? '인식 완료, 전송 중' : '인식 중';
            });
            if (val.finalResult) {
              setState(() {
                _speechStatus = '인식 완료, 계속 듣는 중';
              });
              _pushStatus('음성 인식 결과 업데이트');
            }
          },
        );
      } else {
        setState(() => _speechStatus = '마이크 사용 불가');
        _pushStatus('마이크 사용 불가');
      }
    } else {
      setState(() {
        _isListening = false;
        _speechStatus = '음성 입력 종료, 수정 후 실행 가능';
      });
      _speech.stop();
      _pushStatus('음성 입력 종료');
    }
  }

  Future<void> _sendVoice(String text) async {
    try {
      final res = await http
          .post(
            Uri.parse('$_baseUrl/command/voice'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'text': text}),
          )
          .timeout(const Duration(seconds: 3));
      setState(() => _serverStatus = '명령 전송 ${res.statusCode}');
      _pushStatus(res.statusCode == 200 ? '명령 수신 완료' : '명령 전송 ${res.statusCode}');
      await _fetchLatest();
    } catch (e) {
      setState(() => _serverStatus = '명령 전송 실패: $e');
      _pushStatus('명령 전송 실패');
    }
  }

  Future<void> _sendTypedCommand() async {
    final text = _commandController.text.trim();
    if (text.isEmpty) {
      setState(() => _serverStatus = '텍스트 명령을 입력하세요');
      return;
    }
    await _sendVoice(text);
    _saveAction('confirmed_command', text);
  }

  Future<void> _executeSelectedCommand() async {
    if (_commandController.text.trim().isEmpty) {
      _commandController.text = _basketCommand(_selectedBasket);
    }
    setState(() {
      _robotRunning = true;
      _taskSucceeded = false;
    });
    _pushStatus('$_selectedBasket basket 명령 전송됨');
    await _sendTypedCommand();
  }

  Future<void> _stopRobotUiOnly() async {
    try {
      final res = await http.post(Uri.parse('$_baseUrl/command/stop')).timeout(const Duration(seconds: 3));
      setState(() {
        _robotRunning = false;
        _serverStatus = '정지 요청 ${res.statusCode}';
      });
      _pushStatus(res.statusCode == 200 ? '정지 요청 전송' : '정지 요청 ${res.statusCode}');
      await _fetchLatest();
    } catch (e) {
      setState(() => _serverStatus = '정지 요청 실패: $e');
      _pushStatus('정지 요청 실패');
    }
  }

  void _saveAction(String type, dynamic value) {
    try {
      _dbRef?.child('vla_data').push().set({
        'timestamp': ServerValue.timestamp,
        'type': type,
        'value': value,
      });
    } catch (e) {
      debugPrint('Firebase save skipped: $e');
    }
  }

  Widget _statusChip(String text, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.10),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: color.withValues(alpha: 0.18)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 15, color: color),
          const SizedBox(width: 6),
          Text(text, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w700)),
        ],
      ),
    );
  }

  Widget _viewTab(String id, String label) {
    final selected = _selectedView == id;
    return Expanded(
      child: InkWell(
        borderRadius: BorderRadius.circular(22),
        onTap: () => setState(() => _selectedView = id),
        child: Container(
          height: 42,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: selected ? const Color(0xFF16A34A) : Colors.transparent,
            borderRadius: BorderRadius.circular(22),
          ),
          child: Text(
            label,
            style: TextStyle(color: selected ? Colors.white : const Color(0xFF111827), fontWeight: FontWeight.w700),
          ),
        ),
      ),
    );
  }

  Widget _cameraCard(String label, String url, {bool side = false}) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF111827),
        borderRadius: BorderRadius.circular(8),
        boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.10), blurRadius: 14, offset: const Offset(0, 8))],
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        children: [
          Positioned.fill(
            child: _isStreaming
                ? ClipRect(
                    child: Transform.scale(
                      scaleX: 2.0,
                      scaleY: 1.0,
                      alignment: side ? Alignment.centerRight : Alignment.centerLeft,
                      child: Mjpeg(
                        isLive: true,
                        stream: url,
                        fit: BoxFit.fill,
                        error: (context, error, stack) => const Center(
                          child: Icon(Icons.videocam_off, color: Colors.white54, size: 34),
                        ),
                      ),
                    ),
                  )
                : const Center(child: Text('카메라 스트림 연결 중', style: TextStyle(color: Colors.white54))),
          ),
          Positioned(
            top: 10,
            left: 10,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(color: const Color(0xFF16A34A), borderRadius: BorderRadius.circular(16)),
              child: Text(label, style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w700)),
            ),
          ),
          Positioned(
            top: 13,
            right: 12,
            child: Row(children: [
              Icon(Icons.circle, color: _isStreaming ? const Color(0xFF22C55E) : const Color(0xFF94A3B8), size: 9),
              const SizedBox(width: 5),
              Text(_isStreaming ? 'LIVE' : 'OFF', style: const TextStyle(color: Colors.white, fontSize: 11, fontWeight: FontWeight.w700)),
            ]),
          ),
        ],
      ),
    );
  }

  Widget _basketButton(String color, String label, Color accent) {
    final selected = _selectedBasket == color;
    return Expanded(
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: () => _selectBasket(color),
        child: Container(
          height: 102,
          decoration: BoxDecoration(
            color: selected ? accent.withValues(alpha: 0.10) : Colors.white,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: selected ? accent : const Color(0xFFE5E7EB), width: selected ? 1.5 : 1),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.shopping_basket_outlined, color: accent, size: 31),
              const SizedBox(height: 8),
              Text(label, textAlign: TextAlign.center, style: const TextStyle(fontSize: 15, color: Color(0xFF111827), fontWeight: FontWeight.w700)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionCard(Widget child) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFFE5E7EB)),
        boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 12, offset: const Offset(0, 6))],
      ),
      child: child,
    );
  }

  String _timeText() {
    final now = DateTime.now();
    return '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}:${now.second.toString().padLeft(2, '0')}';
  }

  String _historyPayloadText(Map<String, dynamic> item) {
    final payload = item['payload'];
    if (payload is Map && payload['text'] != null) return payload['text'].toString();
    return payload?.toString() ?? '';
  }

  Widget _historyView() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionCard(Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(children: [
                const Text('제어 기록', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: Color(0xFF111827))),
                const Spacer(),
                IconButton(onPressed: _fetchHistory, icon: const Icon(Icons.refresh, color: Color(0xFF16A34A))),
              ]),
              const SizedBox(height: 4),
              Text('연결 대상: $_baseUrl', style: const TextStyle(color: Color(0xFF64748B), fontSize: 12)),
            ],
          )),
          const SizedBox(height: 14),
          if (_historyLoading)
            const Center(child: Padding(padding: EdgeInsets.all(24), child: CircularProgressIndicator()))
          else if (_commandHistory.isEmpty)
            _sectionCard(const Center(
              child: Padding(
                padding: EdgeInsets.symmetric(vertical: 22),
                child: Text('아직 명령 기록이 없습니다', style: TextStyle(color: Color(0xFF64748B))),
              ),
            ))
          else
            for (final item in _commandHistory)
              Container(
                width: double.infinity,
                margin: const EdgeInsets.only(bottom: 10),
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: const Color(0xFFE5E7EB)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
                        decoration: BoxDecoration(color: const Color(0xFF16A34A).withValues(alpha: 0.10), borderRadius: BorderRadius.circular(14)),
                        child: Text(item['kind']?.toString() ?? 'command', style: const TextStyle(color: Color(0xFF16A34A), fontSize: 12, fontWeight: FontWeight.w800)),
                      ),
                      const Spacer(),
                      Text(item['ts']?.toString() ?? '-', style: const TextStyle(color: Color(0xFF64748B), fontSize: 11)),
                    ]),
                    const SizedBox(height: 10),
                    Text(_historyPayloadText(item), style: const TextStyle(color: Color(0xFF111827), fontSize: 15, fontWeight: FontWeight.w700)),
                  ],
                ),
              ),
        ],
      ),
    );
  }

  Widget _settingsView() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 0, 16, 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionCard(Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(children: [
                const Icon(Icons.arrow_back_ios_new, size: 18, color: Color(0xFF111827)),
                const Expanded(
                  child: Text('설정', textAlign: TextAlign.center, style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: Color(0xFF111827))),
                ),
                const SizedBox(width: 18),
              ]),
              const SizedBox(height: 22),
              const Text('서버 설정', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800, color: Color(0xFF111827))),
              const SizedBox(height: 10),
              Container(
                decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(8), border: Border.all(color: const Color(0xFFE5E7EB))),
                child: Column(children: [
                  TextField(
                    controller: _ipController,
                    decoration: const InputDecoration(labelText: 'IP 주소', border: InputBorder.none, contentPadding: EdgeInsets.symmetric(horizontal: 14, vertical: 14)),
                    keyboardType: TextInputType.number,
                  ),
                  const Divider(height: 1, color: Color(0xFFE5E7EB)),
                  TextField(
                    controller: _portController,
                    decoration: const InputDecoration(labelText: '포트', border: InputBorder.none, contentPadding: EdgeInsets.symmetric(horizontal: 14, vertical: 14)),
                    keyboardType: TextInputType.number,
                  ),
                ]),
              ),
              const SizedBox(height: 10),
              Text('현재 연결 대상: $_baseUrl', style: const TextStyle(color: Color(0xFF64748B), fontSize: 12)),
              const SizedBox(height: 14),
              SizedBox(
                width: double.infinity,
                height: 48,
                child: ElevatedButton.icon(
                  onPressed: _checkServerHealth,
                  icon: const Icon(Icons.wifi, color: Colors.white),
                  label: const Text('서버 확인', style: TextStyle(fontWeight: FontWeight.w800)),
                  style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF16A34A), foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8))),
                ),
              ),
            ],
          )),
          const SizedBox(height: 18),
          const Text('기타', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800, color: Color(0xFF111827))),
          const SizedBox(height: 10),
          _sectionCard(Column(children: const [
            Row(children: [Icon(Icons.help_outline, size: 20, color: Color(0xFF64748B)), SizedBox(width: 10), Text('사용 가이드'), Spacer(), Icon(Icons.chevron_right, color: Color(0xFF64748B))]),
            Divider(height: 24, color: Color(0xFFE5E7EB)),
            Row(children: [Icon(Icons.info_outline, size: 20, color: Color(0xFF64748B)), SizedBox(width: 10), Text('앱 정보'), Spacer(), Icon(Icons.chevron_right, color: Color(0xFF64748B))]),
          ])),
          if (_serverStatus.isNotEmpty) ...[
            const SizedBox(height: 12),
            Text(_serverStatus, style: const TextStyle(color: Color(0xFF64748B), fontSize: 12), maxLines: 2, overflow: TextOverflow.ellipsis),
          ],
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final camUrl = '$_baseUrl/video_feed';
    final command = _commandController.text.trim().isEmpty ? _basketCommand(_selectedBasket) : _commandController.text.trim();

    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 8),
              child: Row(
                children: [
                  const SizedBox(width: 42),
                  const Expanded(
                    child: Text('Robot Control', textAlign: TextAlign.center, style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: Color(0xFF111827))),
                  ),
                  IconButton(onPressed: _checkServerHealth, icon: const Icon(Icons.notifications_none, color: Color(0xFF374151))),
                ],
              ),
            ),
            Expanded(
              child: _selectedTab == 1
                  ? _historyView()
                  : _selectedTab == 2
                      ? _settingsView()
                      : SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 18),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        children: [
                          _statusChip(_isStreaming ? '서버 연결됨' : '서버 대기', Icons.check_circle, const Color(0xFF16A34A)),
                          const SizedBox(width: 8),
                          _statusChip(_isStreaming ? '카메라 정상' : '카메라 대기', Icons.videocam, const Color(0xFF2563EB)),
                          const SizedBox(width: 8),
                          _statusChip(_robotRunning ? '로봇 실행 중' : (_taskSucceeded ? '작업 성공' : '로봇 대기 중'), _taskSucceeded ? Icons.check_circle : Icons.smart_toy, _robotRunning ? const Color(0xFFF59E0B) : (_taskSucceeded ? const Color(0xFF16A34A) : const Color(0xFF64748B))),
                          const SizedBox(width: 8),
                          InkWell(
                            borderRadius: BorderRadius.circular(22),
                            onTap: _toggleStream,
                            child: Container(
                              width: 38,
                              height: 38,
                              alignment: Alignment.center,
                              decoration: BoxDecoration(
                                color: _isStreaming ? const Color(0xFFEF4444) : const Color(0xFF16A34A),
                                borderRadius: BorderRadius.circular(22),
                              ),
                              child: Icon(_isStreaming ? Icons.videocam_off : Icons.videocam, color: Colors.white, size: 19),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 14),
                    Container(
                      padding: const EdgeInsets.all(4),
                      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(24), border: Border.all(color: const Color(0xFFE5E7EB))),
                      child: Row(children: [_viewTab('split', '2분할'), _viewTab('top', 'Top'), _viewTab('side', 'Side')]),
                    ),
                    const SizedBox(height: 14),
                    if (_selectedView == 'split')
                      SizedBox(
                        height: 170,
                        child: Row(children: [
                          Expanded(child: _cameraCard('Top View', camUrl)),
                          const SizedBox(width: 12),
                          Expanded(child: _cameraCard('Side View', camUrl, side: true)),
                        ]),
                      )
                    else
                      SizedBox(
                        height: 240,
                        child: _cameraCard(
                          _selectedView == 'top' ? 'Top View' : 'Side View',
                          camUrl,
                          side: _selectedView == 'side',
                        ),
                      ),
                    const SizedBox(height: 22),
                    const Text('목표 바구니 선택', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800, color: Color(0xFF111827))),
                    const SizedBox(height: 12),
                    Row(children: [
                      _basketButton('green', 'Green\nBasket', const Color(0xFF16A34A)),
                      const SizedBox(width: 12),
                      _basketButton('yellow', 'Yellow\nBasket', const Color(0xFFF59E0B)),
                      const SizedBox(width: 12),
                      _basketButton('blue', 'Blue\nBasket', const Color(0xFF3B82F6)),
                    ]),
                    const SizedBox(height: 16),
                    _sectionCard(Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('선택된 명령', style: TextStyle(fontWeight: FontWeight.w800, color: Color(0xFF111827))),
                        const SizedBox(height: 12),
                        Center(child: Text(command, textAlign: TextAlign.center, style: const TextStyle(color: Color(0xFF059669), fontSize: 18, height: 1.25, fontWeight: FontWeight.w800))),
                        const SizedBox(height: 14),
                        SizedBox(
                          width: double.infinity,
                          height: 48,
                          child: ElevatedButton.icon(
                            onPressed: _executeSelectedCommand,
                            icon: const Icon(Icons.play_arrow, color: Colors.white),
                            label: const Text('실행하기', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800)),
                            style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF16A34A), foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8))),
                          ),
                        ),
                        if (_taskSucceeded) ...[
                          const SizedBox(height: 10),
                          Container(
                            width: double.infinity,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                            decoration: BoxDecoration(color: const Color(0xFFECFDF5), borderRadius: BorderRadius.circular(8), border: Border.all(color: const Color(0xFF86EFAC))),
                            child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [Icon(Icons.check_circle, color: Color(0xFF16A34A)), SizedBox(width: 8), Text('작업을 성공적으로 완료했습니다', style: TextStyle(color: Color(0xFF047857), fontWeight: FontWeight.w800))]),
                          ),
                        ],
                        if (_robotRunning) ...[
                          const SizedBox(height: 10),
                          SizedBox(
                            width: double.infinity,
                            height: 44,
                            child: ElevatedButton.icon(
                              onPressed: _stopRobotUiOnly,
                              icon: const Icon(Icons.stop, color: Colors.white),
                              label: const Text('정지하기', style: TextStyle(fontWeight: FontWeight.w800)),
                              style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFFEF4444), foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8))),
                            ),
                          ),
                        ],
                      ],
                    )),
                    const SizedBox(height: 16),
                    const Text('보조 입력', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w800, color: Color(0xFF111827))),
                    const SizedBox(height: 10),
                    Row(children: [
                      Expanded(child: OutlinedButton.icon(onPressed: _listen, icon: Icon(_isListening ? Icons.mic : Icons.mic_none, color: _isListening ? const Color(0xFFEF4444) : const Color(0xFF16A34A)), label: Text(_isListening ? '듣는 중...' : '음성으로 명령하기'), style: OutlinedButton.styleFrom(foregroundColor: const Color(0xFF374151), side: BorderSide(color: _isListening ? const Color(0xFFEF4444) : const Color(0xFFE5E7EB)), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)), padding: const EdgeInsets.symmetric(vertical: 14)))),
                      const SizedBox(width: 12),
                      Expanded(child: OutlinedButton.icon(onPressed: () {}, icon: const Icon(Icons.keyboard, color: Color(0xFF64748B)), label: const Text('텍스트 직접 입력'), style: OutlinedButton.styleFrom(foregroundColor: const Color(0xFF374151), side: const BorderSide(color: Color(0xFFE5E7EB)), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)), padding: const EdgeInsets.symmetric(vertical: 14)))),
                    ]),
                    const SizedBox(height: 8),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                      decoration: BoxDecoration(
                        color: _isListening ? const Color(0xFFFFF1F2) : Colors.white,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: _isListening ? const Color(0xFFEF4444) : const Color(0xFFE5E7EB)),
                      ),
                      child: Row(children: [
                        Icon(_isListening ? Icons.graphic_eq : Icons.mic_none, color: _isListening ? const Color(0xFFEF4444) : const Color(0xFF64748B), size: 18),
                        const SizedBox(width: 8),
                        Expanded(child: Text('$_speechStatus · $_text', style: const TextStyle(color: Color(0xFF374151), fontSize: 12), maxLines: 1, overflow: TextOverflow.ellipsis)),
                      ]),
                    ),
                    const SizedBox(height: 10),
                    TextField(
                      controller: _commandController,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (_) => _sendTypedCommand(),
                      decoration: InputDecoration(
                        hintText: '직접 명령 입력',
                        filled: true,
                        fillColor: Colors.white,
                        suffixIcon: IconButton(icon: const Icon(Icons.send, color: Color(0xFF16A34A)), onPressed: _sendTypedCommand),
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: const BorderSide(color: Color(0xFFE5E7EB))),
                        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: const BorderSide(color: Color(0xFFE5E7EB))),
                      ),
                    ),
                    const SizedBox(height: 16),
                    _sectionCard(Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(children: [
                          const Text('최근 상태', style: TextStyle(fontWeight: FontWeight.w800, color: Color(0xFF111827))),
                          const Spacer(),
                          TextButton(onPressed: _fetchLatest, child: const Text('더보기 >', style: TextStyle(color: Color(0xFF64748B)))),
                        ]),
                        const SizedBox(height: 4),
                        for (final item in _statusHistory.take(3))
                          Padding(
                            padding: const EdgeInsets.symmetric(vertical: 5),
                            child: Row(children: [
                              Text(_timeText(), style: const TextStyle(color: Color(0xFF64748B), fontSize: 13)),
                              const SizedBox(width: 18),
                              Expanded(child: Text(item, style: const TextStyle(color: Color(0xFF374151), fontSize: 13))),
                              const Icon(Icons.circle, color: Color(0xFF16A34A), size: 8),
                            ]),
                          ),
                        if (_latestInstruction != '(없음)')
                          Text(_latestInstruction, style: const TextStyle(color: Color(0xFF059669), fontSize: 12), maxLines: 2, overflow: TextOverflow.ellipsis),
                        if (_latestCommand != '(없음)')
                          Text(_latestCommand, style: const TextStyle(color: Color(0xFF64748B), fontSize: 11), maxLines: 1, overflow: TextOverflow.ellipsis),
                      ],
                    )),
                    if (_serverStatus.isNotEmpty) ...[
                      const SizedBox(height: 10),
                      Text(_serverStatus, style: const TextStyle(color: Color(0xFF64748B), fontSize: 11), maxLines: 2, overflow: TextOverflow.ellipsis),
                    ],
                  ],
                ),
              ),
            ),
            Container(
              height: 66,
              decoration: const BoxDecoration(color: Colors.white, border: Border(top: BorderSide(color: Color(0xFFE5E7EB)))),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _BottomNavItem(icon: Icons.home, label: '홈', active: _selectedTab == 0, onTap: () => _selectTab(0)),
                  _BottomNavItem(icon: Icons.article_outlined, label: '제어 기록', active: _selectedTab == 1, onTap: () => _selectTab(1)),
                  _BottomNavItem(icon: Icons.settings_outlined, label: '설정', active: _selectedTab == 2, onTap: () => _selectTab(2)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _BottomNavItem extends StatelessWidget {
  const _BottomNavItem({required this.icon, required this.label, this.active = false, this.onTap});

  final IconData icon;
  final String label;
  final bool active;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final color = active ? const Color(0xFF16A34A) : const Color(0xFF64748B);
    return InkWell(
      borderRadius: BorderRadius.circular(8),
      onTap: onTap,
      child: SizedBox(
        width: 90,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: color, size: 23),
            const SizedBox(height: 3),
            Text(label, style: TextStyle(color: color, fontSize: 11, fontWeight: active ? FontWeight.w800 : FontWeight.w500)),
          ],
        ),
      ),
    );
  }
}
