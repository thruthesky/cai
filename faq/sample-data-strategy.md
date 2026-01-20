# 샘플 데이터 구성 전략

JAI LLM 학습을 위한 샘플 데이터를 어떻게 구성해야 하는지 상세하게 설명합니다.

---

## 개요

JAI는 구인 정보 데이터베이스를 기반으로 학습 데이터를 생성합니다. 핵심은 **"질문 → 문서 참고 → 답변"** 패턴을 모델에게 학습시키는 것입니다.

```
사용자 질문 + 원본 구인 정보 → 구조화된 답변
```

---

## 1. 학습 데이터 기본 구조

JAI의 학습 데이터는 `[QUESTION]/[DOC]/[ANSWER]` 세 부분으로 구성됩니다.

```
[QUESTION]
사용자 질문
[/QUESTION]

[DOC]
원본 구인 정보 (DB 필드 기반)
[/DOC]

[ANSWER]
요약:
- ...

체크리스트:
- ...

구인 정보:
- ...

상세 설명:
...
[/ANSWER]
```

### 각 섹션의 역할

| 섹션 | 역할 | 예시 |
|------|------|------|
| `[QUESTION]` | 사용자가 물어볼 수 있는 질문 | "서울에서 React 개발자 채용 있어?" |
| `[DOC]` | 답변의 근거가 되는 원본 데이터 | DB에서 가져온 구인 공고 정보 |
| `[ANSWER]` | 모델이 출력해야 할 구조화된 답변 | 요약, 체크리스트, 구인 정보, 상세 설명 |

### [DOC]가 있는 이유

```
[DOC]가 없으면: 모델이 답변을 "암기" → 새로운 데이터에 대응 불가
[DOC]가 있으면: 모델이 "문서를 참고해서 답변하는 방법"을 학습 → RAG 연동에 유리
```

모델은 이 형식을 수천 번 보면서 **"질문이 들어오면 문서를 참고해서 구조화된 답변을 만든다"**는 패턴을 학습합니다.

---

## 2. 구인 정보 데이터베이스 필드

JAI가 사용하는 구인 정보 DB의 필드 구조입니다.

### 기본 정보

| 필드 | 설명 | 예시 |
|------|------|------|
| `title` | 구인 공고 제목 | "[서울 강남] 프론트엔드 개발자 채용" |
| `company_name` | 채용 회사명 | "테크스타트업 주식회사" |
| `position` | 모집 직책/포지션 | "프론트엔드 개발자" |
| `department` | 소속 부서 | "개발팀" |
| `industry` | 회사 업종 | IT, 제조, 서비스, 금융, 교육, 의료, 유통, 건설, 미디어, 기타 |

### 근무 조건

| 필드 | 설명 | 예시/값 |
|------|------|--------|
| `location` | 근무 장소 | "서울시 강남구 테헤란로 123" |
| `employment_type` | 고용 형태 | 정규직, 계약직, 파트타임, 인턴, 프리랜서 |
| `remote_work` | 원격 근무 가능 여부 | no(불가), yes(가능), hybrid(하이브리드) |
| `work_hours` | 근무 시간 | "09:00-18:00" |
| `work_days` | 근무 요일 | "월-금" |
| `hiring_count` | 채용 인원 | "2명" |
| `probation_period` | 수습 기간 | "3개월" |
| `team_size` | 팀 규모 | "5-10명" |

### 급여 정보

| 필드 | 설명 | 예시/값 |
|------|------|--------|
| `salary_type` | 급여 지급 방식 | annual(연봉), monthly(월급), hourly(시급), daily(일급) |
| `salary_currency` | 급여 통화 | KRW, USD, JPY, EUR, CNY |
| `salary_negotiable` | 급여 협상 가능 | yes(협상 가능), no(협상 불가) |
| `salary` | 급여 범위 텍스트 | "5000만원~7000만원" |
| `min_salary` | 최소 급여 (숫자) | 50000000 |
| `max_salary` | 최대 급여 (숫자) | 70000000 |

### 경력 요건

| 필드 | 설명 | 예시/값 |
|------|------|--------|
| `experience_level` | 경력 수준 | 신입, 경력, 무관 |
| `min_experience_years` | 최소 경력 연차 | 0, 3, 5 |
| `max_experience_years` | 최대 경력 연차 | 3, 10, 15 |
| `education_level` | 학력 | 학력무관, 고졸, 초대졸, 대졸, 석사, 박사 |
| `languages` | 필요 언어 | "영어 비즈니스급" |

### 채용 일정

| 필드 | 설명 | 예시 |
|------|------|------|
| `deadline` | 지원 마감일 | "2026-02-28" |
| `start_date` | 입사 시작일 | "2026-03-15" |

### 상세 정보

| 필드 | 설명 | 예시 |
|------|------|------|
| `responsibilities` | 주요 업무 | "웹 프론트엔드 개발\nAPI 연동" |
| `requirements` | 자격 요건 | "[필수] React 3년 이상\n[우대] TypeScript" |
| `skills` | 기술 스택 | "React, TypeScript, Node.js" |
| `certifications` | 자격증 | "정보처리기사" |
| `content` | 추가 설명 | "우리 회사는..." |

### 복리후생

| 필드 | 설명 | 예시 |
|------|------|------|
| `benefits` | 복리후생 | "4대보험\n연차 15일\n점심 식대" |
| `visa_sponsorship` | 비자 지원 | yes(지원), no(미지원) |

### 지원 방법

| 필드 | 설명 | 예시 |
|------|------|------|
| `application_url` | 지원 페이지 URL | "https://company.com/apply" |
| `application_email` | 지원 이메일 | "hr@company.com" |
| `contact_person` | 채용 담당자 | "김인사" |
| `contact_phone` | 담당자 연락처 | "010-1234-5678" |
| `application_docs` | 제출 서류 | "이력서, 포트폴리오" |

### 필수 vs 선택 필드

**필수 필드**: `title`, `company_name`, `position`, `industry`, `location`, `employment_type`, `deadline`, `responsibilities`, `application_email`, `contact_person`, `contact_phone`

**선택 필드**: 나머지 모든 필드

---

## 3. [DOC] 섹션 변환 규칙

DB 레코드를 [DOC] 형식으로 변환할 때의 규칙입니다.

### 변환 예시

**DB 레코드:**
```json
{
  "title": "[서울 강남] 프론트엔드 개발자 채용",
  "company_name": "테크스타트업 주식회사",
  "position": "프론트엔드 개발자",
  "industry": "IT",
  "location": "서울시 강남구 테헤란로 123",
  "employment_type": "정규직",
  "remote_work": "hybrid",
  "work_hours": "09:00-18:00",
  "work_days": "월-금",
  "hiring_count": "2명",
  "probation_period": "3개월",
  "salary_type": "annual",
  "salary": "5000만원~7000만원",
  "salary_negotiable": "yes",
  "experience_level": "경력",
  "min_experience_years": "3",
  "max_experience_years": "7",
  "education_level": "대졸",
  "languages": "영어 비즈니스급",
  "deadline": "2026-02-28",
  "start_date": "2026-03-15",
  "responsibilities": "웹 프론트엔드 개발 및 유지보수\nReact/Vue.js 기반 UI 구현\n백엔드 API 연동",
  "requirements": "[필수]\n- React 경험 3년 이상\n- TypeScript 능숙\n[우대]\n- 핀테크 도메인 경험",
  "skills": "React, TypeScript, Node.js, PostgreSQL",
  "certifications": "AWS Solutions Architect (우대)",
  "benefits": "4대보험\n연차 15일\n점심 식대 지원\n자기계발비 월 10만원",
  "application_email": "hr@techstartup.com",
  "contact_person": "김인사",
  "contact_phone": "010-1234-5678",
  "application_docs": "이력서, 포트폴리오"
}
```

**[DOC] 변환 결과:**
```
[DOC]
제목: [서울 강남] 프론트엔드 개발자 채용
회사: 테크스타트업 주식회사
업종: IT

포지션: 프론트엔드 개발자
채용인원: 2명

근무지: 서울시 강남구 테헤란로 123
고용형태: 정규직
원격근무: 하이브리드
근무시간: 09:00-18:00 (월-금)
수습기간: 3개월

급여: 5000만원~7000만원 (연봉, 협상 가능)

경력: 경력 3~7년
학력: 대졸
언어: 영어 비즈니스급

마감일: 2026-02-28
입사예정일: 2026-03-15

주요업무:
- 웹 프론트엔드 개발 및 유지보수
- React/Vue.js 기반 UI 구현
- 백엔드 API 연동

자격요건:
[필수]
- React 경험 3년 이상
- TypeScript 능숙
[우대]
- 핀테크 도메인 경험

기술스택: React, TypeScript, Node.js, PostgreSQL
자격증: AWS Solutions Architect (우대)

복리후생:
- 4대보험
- 연차 15일
- 점심 식대 지원
- 자기계발비 월 10만원

지원방법:
- 이메일: hr@techstartup.com
- 담당자: 김인사 (010-1234-5678)
- 제출서류: 이력서, 포트폴리오
[/DOC]
```

### 필드별 변환 규칙

| DB 값 | [DOC] 표현 |
|-------|-----------|
| `salary_type: annual` | "연봉" |
| `salary_type: monthly` | "월급" |
| `salary_type: hourly` | "시급" |
| `remote_work: no` | "불가" |
| `remote_work: yes` | "가능" |
| `remote_work: hybrid` | "하이브리드" |
| `salary_negotiable: yes` | "(협상 가능)" 추가 |
| `visa_sponsorship: yes` | "비자 지원: 가능" 추가 |
| `min_experience_years` + `max_experience_years` | "경력 3~7년" |
| `min_experience_years`만 있을 때 | "경력 3년 이상" |

---

## 4. [ANSWER] 섹션 구성

모델이 출력할 구조화된 응답 형식입니다.

### 표준 형식

```
[ANSWER]
요약:
- 핵심 정보 1~3줄

체크리스트:
- 지원 자격:
  - (1) 필수 조건 1
  - (2) 필수 조건 2
- 우대 사항:
  - (1) 우대 조건 1
- 준비물:
  - (1) 제출 서류 1
  - (2) 제출 서류 2

구인 정보:
- 회사명
  - 포지션: ...
  - 연봉: ...
  - 위치: ...
  - 고용형태: ...
  - 원격근무: ...
  - 채용인원: ...
  - 마감: ...
  - 연락처: ...

상세 설명:
자연어로 된 상세 설명 (2~3문장)
[/ANSWER]
```

### 변환 예시

위의 [DOC]에 대한 [ANSWER]:

```
[ANSWER]
요약:
- 서울 강남 IT 스타트업에서 프론트엔드 개발자 채용 중
- 연봉 5000~7000만원, 하이브리드 근무 가능
- 마감일: 2026-02-28

체크리스트:
- 지원 자격:
  - (1) React 경험 3년 이상
  - (2) TypeScript 능숙
  - (3) 대졸 이상
- 우대 사항:
  - (1) 핀테크 도메인 경험
  - (2) AWS 자격증
- 준비물:
  - (1) 이력서
  - (2) 포트폴리오

구인 정보:
- 테크스타트업 주식회사
  - 포지션: 프론트엔드 개발자
  - 연봉: 5000만원 ~ 7000만원 (협상 가능)
  - 위치: 서울시 강남구 테헤란로 123
  - 고용형태: 정규직 (수습 3개월)
  - 원격근무: 하이브리드
  - 채용인원: 2명
  - 마감: 2026-02-28
  - 연락처: hr@techstartup.com (김인사)

상세 설명:
테크스타트업은 혁신적인 핀테크 서비스를 제공하는 회사로, React/TypeScript 기반 프론트엔드 개발자를 찾고 있습니다. 월-금 09:00-18:00 근무이며, 하이브리드 원격근무가 가능합니다. 4대보험, 연차 15일, 점심 식대 지원, 자기계발비 월 10만원 등의 복리후생이 제공됩니다.
[/ANSWER]
```

---

## 5. 질문 유형별 샘플

다양한 검색 패턴을 학습시키기 위해 **하나의 구인 정보로 여러 질문**을 만듭니다.

### 질문 유형 분류

| 유형 | 활용 필드 | 예시 질문 |
|------|----------|----------|
| 지역 기반 | `location` | "서울 강남에서 채용 중인 회사 있어?" |
| 직종 기반 | `position`, `industry` | "프론트엔드 개발자 구인 있어?" |
| 회사 기반 | `company_name` | "테크스타트업 채용 정보 알려줘" |
| 급여 기반 | `salary`, `min_salary` | "연봉 6천만원 이상 개발자 채용 있어?" |
| 경력 기반 | `experience_level` | "신입 개발자 채용 있어?" |
| 기술 스택 | `skills` | "React 개발자 채용 있어?" |
| 근무 형태 | `employment_type`, `remote_work` | "재택근무 가능한 IT 회사 있어?" |
| 학력 기반 | `education_level` | "학력 무관 개발자 채용 있어?" |
| 복합 조건 | 여러 필드 | "서울에서 React 개발자 정규직 연봉 5천 이상" |

### 질문 생성 예시

같은 구인 정보로 만들 수 있는 질문들:

```
# 지역 기반
서울에서 프론트엔드 개발자 채용 있어?
강남구에서 IT 회사 채용 있어?

# 회사 기반
테크스타트업 주식회사 채용 정보 알려줘
테크스타트업에서 뭐 뽑아?

# 급여 기반
연봉 5천만원 이상 개발자 채용 있어?
연봉 7천까지 주는 프론트엔드 채용 있어?

# 경력 기반
경력 3년차 프론트엔드 개발자 채용 있어?
경력 5년 이상 필요한 채용 있어?

# 기술 스택 기반
React 개발자 채용 있어?
TypeScript 쓰는 회사 채용 있어?

# 근무 형태 기반
재택근무 가능한 프론트엔드 채용 있어?
하이브리드 근무 가능한 IT 채용 있어?

# 복합 조건
서울 강남에서 React 개발자 정규직 연봉 5천 이상 채용 있어?
재택 가능하고 연봉 6천 이상인 프론트엔드 채용 있어?
```

---

## 6. 다중 결과 응답

여러 구인 정보가 매칭될 때의 응답 형식입니다.

### [DOC]에 여러 공고 포함

```
[QUESTION]
서울에서 프론트엔드 개발자 채용 있어?
[/QUESTION]

[DOC]
--- 공고 1 ---
제목: [서울 강남] 프론트엔드 개발자 채용
회사: 테크스타트업 주식회사
...

--- 공고 2 ---
제목: [서울 판교] React 개발자 모집
회사: 핀테크코리아
...

--- 공고 3 ---
제목: [서울 성수] 웹 프론트엔드 엔지니어
회사: 이커머스랩
...
[/DOC]

[ANSWER]
요약:
- 서울에서 프론트엔드 개발자 채용 3건 발견
- 강남, 판교, 성수 지역에서 채용 진행 중
- 연봉 범위: 5000만원 ~ 9000만원

체크리스트:
- 공통 자격:
  - (1) React 경험 필수
  - (2) JavaScript/TypeScript 능숙
- 추가 우대:
  - (1) 핀테크 경험 (테크스타트업, 핀테크코리아)
  - (2) 이커머스 경험 (이커머스랩)

구인 정보:
- 테크스타트업 주식회사
  - 포지션: 프론트엔드 개발자
  - 연봉: 5000만원 ~ 7000만원
  - 위치: 서울 강남구
  - 마감: 2026-02-28

- 핀테크코리아
  - 포지션: React 개발자
  - 연봉: 6000만원 ~ 8000만원
  - 위치: 서울 판교
  - 마감: 2026-03-15

- 이커머스랩
  - 포지션: 웹 프론트엔드 엔지니어
  - 연봉: 7000만원 ~ 9000만원
  - 위치: 서울 성수
  - 마감: 2026-02-20

상세 설명:
서울 지역에서 프론트엔드 개발자를 채용하는 3개 회사를 찾았습니다. 테크스타트업은 강남에 위치하며 하이브리드 근무가 가능합니다. 핀테크코리아는 판교에 위치하며 핀테크 도메인 경험자를 우대합니다. 이커머스랩은 성수에 위치하며 가장 높은 연봉을 제시하고 있습니다.
[/ANSWER]
```

---

## 7. 변환 스크립트 구현

`scripts/prepare_samples.py`에서 사용할 변환 함수들입니다.

### job_to_doc 함수

```python
def job_to_doc(job: dict) -> str:
    """DB 레코드를 [DOC] 형식으로 변환"""
    lines = []

    # 기본 정보
    lines.append(f"제목: {job['title']}")
    lines.append(f"회사: {job['company_name']}")
    lines.append(f"업종: {job['industry']}")
    lines.append("")

    # 포지션 정보
    lines.append(f"포지션: {job['position']}")
    if job.get('department'):
        lines.append(f"부서: {job['department']}")
    if job.get('hiring_count'):
        lines.append(f"채용인원: {job['hiring_count']}")
    lines.append("")

    # 근무 조건
    lines.append(f"근무지: {job['location']}")
    lines.append(f"고용형태: {job['employment_type']}")

    # 원격근무 표현
    remote_map = {'no': '불가', 'yes': '가능', 'hybrid': '하이브리드'}
    if job.get('remote_work'):
        lines.append(f"원격근무: {remote_map.get(job['remote_work'], job['remote_work'])}")

    # 근무시간
    if job.get('work_hours') and job.get('work_days'):
        lines.append(f"근무시간: {job['work_hours']} ({job['work_days']})")

    # 수습기간
    if job.get('probation_period'):
        lines.append(f"수습기간: {job['probation_period']}")
    lines.append("")

    # 급여 정보
    salary_text = job.get('salary', '')
    salary_type_map = {
        'annual': '연봉',
        'monthly': '월급',
        'hourly': '시급',
        'daily': '일급'
    }
    salary_type = salary_type_map.get(job.get('salary_type', ''), '')
    negotiable = ", 협상 가능" if job.get('salary_negotiable') == 'yes' else ""
    if salary_text:
        lines.append(f"급여: {salary_text} ({salary_type}{negotiable})")
    lines.append("")

    # 경력/학력 요건
    exp_text = job.get('experience_level', '')
    if job.get('min_experience_years') and job.get('max_experience_years'):
        exp_text += f" {job['min_experience_years']}~{job['max_experience_years']}년"
    elif job.get('min_experience_years'):
        exp_text += f" {job['min_experience_years']}년 이상"
    if exp_text:
        lines.append(f"경력: {exp_text}")

    if job.get('education_level'):
        lines.append(f"학력: {job['education_level']}")
    if job.get('languages'):
        lines.append(f"언어: {job['languages']}")
    lines.append("")

    # 일정
    lines.append(f"마감일: {job['deadline']}")
    if job.get('start_date'):
        lines.append(f"입사예정일: {job['start_date']}")
    lines.append("")

    # 상세 정보
    if job.get('responsibilities'):
        lines.append("주요업무:")
        for resp in job['responsibilities'].split('\n'):
            if resp.strip():
                lines.append(f"- {resp.strip()}")
        lines.append("")

    if job.get('requirements'):
        lines.append("자격요건:")
        lines.append(job['requirements'])
        lines.append("")

    if job.get('skills'):
        lines.append(f"기술스택: {job['skills']}")

    if job.get('certifications'):
        lines.append(f"자격증: {job['certifications']}")
    if job.get('skills') or job.get('certifications'):
        lines.append("")

    # 복리후생
    if job.get('benefits'):
        lines.append("복리후생:")
        for benefit in job['benefits'].split('\n'):
            if benefit.strip():
                lines.append(f"- {benefit.strip()}")
        lines.append("")

    if job.get('visa_sponsorship') == 'yes':
        lines.append("비자 지원: 가능")
        lines.append("")

    # 지원 방법
    lines.append("지원방법:")
    if job.get('application_email'):
        lines.append(f"- 이메일: {job['application_email']}")
    if job.get('contact_person'):
        contact = job['contact_person']
        if job.get('contact_phone'):
            contact += f" ({job['contact_phone']})"
        lines.append(f"- 담당자: {contact}")
    if job.get('application_docs'):
        lines.append(f"- 제출서류: {job['application_docs']}")
    if job.get('application_url'):
        lines.append(f"- 지원페이지: {job['application_url']}")

    return '\n'.join(lines)
```

### generate_questions 함수

```python
def generate_questions(job: dict) -> list[str]:
    """하나의 구인 정보로 여러 질문 생성"""
    questions = []

    # 지역 추출 (예: "서울시 강남구" → "서울", "강남")
    location = job['location']
    city = location.split()[0].replace('시', '')  # "서울"
    district = location.split()[1] if len(location.split()) > 1 else ""  # "강남구"

    # 지역 기반 질문
    questions.append(f"{city}에서 {job['position']} 채용 있어?")
    if district:
        questions.append(f"{district}에서 {job['industry']} 회사 채용 있어?")

    # 회사 기반 질문
    questions.append(f"{job['company_name']} 채용 정보 알려줘")

    # 급여 기반 질문
    if job.get('min_salary'):
        salary_man = int(job['min_salary']) // 10000  # 원 → 만원
        questions.append(f"연봉 {salary_man}만원 이상 {job['industry']} 채용 있어?")

    # 경력 기반 질문
    if job['experience_level'] == '신입':
        questions.append(f"신입 {job['position']} 채용 있어?")
    elif job['experience_level'] == '경력':
        if job.get('min_experience_years'):
            questions.append(f"경력 {job['min_experience_years']}년차 {job['position']} 채용 있어?")

    # 기술 스택 기반 질문
    if job.get('skills'):
        skills = [s.strip() for s in job['skills'].split(',')]
        for skill in skills[:2]:  # 상위 2개 기술만
            questions.append(f"{skill} 개발자 채용 있어?")

    # 근무 형태 기반 질문
    if job.get('remote_work') in ['yes', 'hybrid']:
        questions.append(f"재택근무 가능한 {job['position']} 채용 있어?")

    # 복합 조건 질문
    if job.get('min_salary') and job.get('skills'):
        skill = job['skills'].split(',')[0].strip()
        salary_man = int(job['min_salary']) // 10000
        questions.append(f"{city}에서 {skill} 개발자 연봉 {salary_man}만원 이상 채용 있어?")

    return questions
```

### job_to_answer 함수

```python
def job_to_answer(job: dict) -> str:
    """DB 레코드를 [ANSWER] 형식으로 변환"""
    lines = []

    # 요약
    lines.append("요약:")
    city = job['location'].split()[0].replace('시', '')
    lines.append(f"- {city} {job['industry']} 회사에서 {job['position']} 채용 중")
    if job.get('salary'):
        remote = ""
        if job.get('remote_work') == 'hybrid':
            remote = ", 하이브리드 근무 가능"
        elif job.get('remote_work') == 'yes':
            remote = ", 완전 원격근무 가능"
        lines.append(f"- {job['salary']}{remote}")
    lines.append(f"- 마감일: {job['deadline']}")
    lines.append("")

    # 체크리스트
    lines.append("체크리스트:")

    # 지원 자격
    lines.append("- 지원 자격:")
    req_num = 1
    if job.get('requirements'):
        for req in job['requirements'].split('\n'):
            req = req.strip()
            if req and not req.startswith('[') and req.startswith('-'):
                lines.append(f"  - ({req_num}) {req[1:].strip()}")
                req_num += 1
    if job.get('education_level') and job['education_level'] != '학력무관':
        lines.append(f"  - ({req_num}) {job['education_level']} 이상")
        req_num += 1

    # 우대 사항 (requirements에서 [우대] 섹션 추출)
    if job.get('requirements') and '[우대]' in job['requirements']:
        lines.append("- 우대 사항:")
        pref_num = 1
        in_pref = False
        for req in job['requirements'].split('\n'):
            req = req.strip()
            if '[우대]' in req:
                in_pref = True
                continue
            if in_pref and req.startswith('-'):
                lines.append(f"  - ({pref_num}) {req[1:].strip()}")
                pref_num += 1

    # 준비물
    if job.get('application_docs'):
        lines.append("- 준비물:")
        doc_num = 1
        for doc in job['application_docs'].split(','):
            lines.append(f"  - ({doc_num}) {doc.strip()}")
            doc_num += 1
    lines.append("")

    # 구인 정보
    lines.append("구인 정보:")
    lines.append(f"- {job['company_name']}")
    lines.append(f"  - 포지션: {job['position']}")
    if job.get('salary'):
        negotiable = " (협상 가능)" if job.get('salary_negotiable') == 'yes' else ""
        lines.append(f"  - 연봉: {job['salary']}{negotiable}")
    lines.append(f"  - 위치: {job['location']}")
    probation = f" (수습 {job['probation_period']})" if job.get('probation_period') else ""
    lines.append(f"  - 고용형태: {job['employment_type']}{probation}")
    if job.get('remote_work'):
        remote_map = {'no': '불가', 'yes': '완전 원격', 'hybrid': '하이브리드'}
        lines.append(f"  - 원격근무: {remote_map.get(job['remote_work'], '')}")
    if job.get('hiring_count'):
        lines.append(f"  - 채용인원: {job['hiring_count']}")
    lines.append(f"  - 마감: {job['deadline']}")
    if job.get('application_email'):
        contact = job['application_email']
        if job.get('contact_person'):
            contact += f" ({job['contact_person']})"
        lines.append(f"  - 연락처: {contact}")
    lines.append("")

    # 상세 설명
    lines.append("상세 설명:")
    desc_parts = []
    desc_parts.append(f"{job['company_name']}에서 {job['position']}을(를) 채용합니다.")
    if job.get('work_hours') and job.get('work_days'):
        desc_parts.append(f"{job['work_days']} {job['work_hours']} 근무입니다.")
    if job.get('remote_work') == 'hybrid':
        desc_parts.append("하이브리드 원격근무가 가능합니다.")
    elif job.get('remote_work') == 'yes':
        desc_parts.append("완전 원격근무가 가능합니다.")
    if job.get('benefits'):
        benefits_list = [b.strip() for b in job['benefits'].split('\n') if b.strip()]
        if len(benefits_list) > 0:
            desc_parts.append(f"{', '.join(benefits_list[:3])} 등의 복리후생이 제공됩니다.")
    lines.append(' '.join(desc_parts))

    return '\n'.join(lines)
```

### 전체 샘플 생성 함수

```python
def generate_sample(job: dict, question: str) -> str:
    """하나의 질문-답변 쌍 생성"""
    doc = job_to_doc(job)
    answer = job_to_answer(job)

    return f"""[QUESTION]
{question}
[/QUESTION]

[DOC]
{doc}
[/DOC]

[ANSWER]
{answer}
[/ANSWER]
"""


def generate_all_samples(jobs: list[dict]) -> list[str]:
    """모든 구인 정보에서 샘플 생성"""
    samples = []

    for job in jobs:
        questions = generate_questions(job)
        for question in questions:
            sample = generate_sample(job, question)
            samples.append(sample)

    return samples
```

---

## 8. 권장 샘플 수

| 데이터 규모 | 구인 공고 수 | 질문-답변 쌍 | 예상 토큰 |
|------------|------------|-------------|----------|
| 초기 테스트 | 100개 | 300~500개 | ~500K |
| 중간 학습 | 1,000개 | 3,000~5,000개 | ~5M |
| 본격 학습 | 10,000개 | 30,000~50,000개 | ~50M |

### 질문 유형 분포 (권장)

모든 유형이 균형 있게 분포되도록 합니다:

| 질문 유형 | 비율 |
|----------|------|
| 지역 기반 | 20% |
| 직종 기반 | 20% |
| 급여 기반 | 15% |
| 기술 스택 | 15% |
| 경력 기반 | 10% |
| 근무 형태 | 10% |
| 복합 조건 | 10% |

---

## 9. 데이터 품질 체크리스트

샘플 데이터 생성 시 확인해야 할 항목:

### 필수 체크

- [ ] 필수 필드가 모두 채워져 있는가?
- [ ] 마감일이 지나지 않은 공고만 포함했는가?
- [ ] 급여 단위가 일관적인가? (원 vs 만원)
- [ ] [DOC]와 [ANSWER]의 정보가 일치하는가?

### 품질 체크

- [ ] 질문 유형이 균형 있게 분포되어 있는가?
- [ ] 다양한 업종이 포함되어 있는가?
- [ ] 다양한 지역이 포함되어 있는가?
- [ ] 신입/경력 비율이 적절한가?

### 형식 체크

- [ ] 모든 태그가 올바르게 열리고 닫혔는가?
- [ ] 줄바꿈이 일관적인가?
- [ ] 특수문자가 이스케이프 없이 포함되어 있지 않은가?

---

## 참고

- [샘플 데이터 요구사항](sample-data.md) - 왜 실제 구인 정보가 필요한가
- [데이터 흐름](data-flow.md) - 전체 파이프라인 설명
- [데이터 준비](../docs/03-data-preparation.md) - 상세 기술 문서
