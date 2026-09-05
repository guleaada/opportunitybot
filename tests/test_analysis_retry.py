import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import database as db
import scorer
import checker
from model_router import extract_json
from analysis_response import AnalysisResponseError


def reply(content, tokens=20):
    return dict(content=content, tokens_used={'output': tokens},
                model_used='stub', cost_usd=0)


class RetryTests(unittest.TestCase):
    def test_truncated_outer_object_never_returns_nested_breakdown(self):
        self.assertIsNone(extract_json('{"overall_score":8,"breakdown":{"funding":3},"sub_scores":{'))
        self.assertEqual(extract_json('Answer: {"overall_score":8}'), {'overall_score':8})
        self.assertEqual(extract_json('```json\n{"a":1}\n```'), {'a':1})
        self.assertEqual(extract_json('[{"a":1}]'), [{'a':1}])

    def test_scoring_retries_and_recovers(self):
        with patch.object(scorer, 'call_model', side_effect=[reply('{"breakdown":'), reply('{"overall_score":8,"funding":"fully_funded"}')]) as call:
            result = scorer.score_opportunity({}, {})
        self.assertEqual(result['overall_score'], 8)
        self.assertEqual(call.call_count, 2)
        self.assertGreater(call.call_args_list[1].kwargs['max_tokens'], call.call_args_list[0].kwargs['max_tokens'])

    def test_invalid_scoring_is_failure_not_zero(self):
        for content in ('{}', '{"funding":3}', '{"overall_score":null}', '{"overall_score":NaN}'):
            with self.subTest(content=content), patch.object(scorer, 'call_model', return_value=reply(content)):
                with self.assertRaises(AnalysisResponseError):
                    scorer.score_opportunity({}, {})

    def test_eligibility_invalid_retries_but_real_negative_stands(self):
        with patch.object(checker, 'call_model', return_value=reply('{}')):
            with self.assertRaises(AnalysisResponseError):
                checker.check_eligibility('text', {})
        with patch.object(checker, 'call_model', return_value=reply('{"eligibility_status":"CONFIRMED_INELIGIBLE","reasoning":"citizenship restriction"}')):
            self.assertEqual(checker.check_eligibility('text', {})['eligibility_status'], 'CONFIRMED_INELIGIBLE')

    def test_recovery_preserves_audit_and_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(db,'SEEN_PATH',Path(tmp)/'seen.json'), patch.object(db,'WATCHLIST_PATH',Path(tmp)/'wl.json'):
            bad=db.save_opportunity({'url':'https://example.org/a','status':'analyzed','score':0,'reward':3})
            good=db.save_opportunity({'url':'https://example.org/b','status':'analyzed','score':0,'reward':'none'})
            self.assertEqual(db.recover_failed_analyses(),1)
            self.assertEqual(db.recover_failed_analyses(),0)
            self.assertIn(bad,db.all_seen())
            self.assertIn(bad,db.all_watchlist())
            self.assertNotIn(good,db.all_watchlist())

class WatchlistTests(unittest.TestCase):
    def test_failed_analysis_stays_queued_and_success_removes_it(self):
        import main
        from analysis_response import AnalysisResponseError
        with tempfile.TemporaryDirectory() as tmp, patch.object(db,'SEEN_PATH',Path(tmp)/'seen.json'), patch.object(db,'WATCHLIST_PATH',Path(tmp)/'wl.json'), patch.object(main, 'log_run_metadata'), patch.object(main.tools, 'send_notification'):
            oid = db.add_to_watchlist({'url':'https://example.org/retry','title':'Retry','retry_analysis':True})
            with patch.object(main, '_analyze_one_inner', side_effect=AnalysisResponseError('invalid JSON')):
                main.check_watchlist()
            self.assertIn(oid, db.all_watchlist())
            self.assertNotIn(oid, db.all_seen())
            self.assertIn('last_checked', db.all_watchlist()[oid])
            def terminal(result, stats):
                db.save_opportunity({'url':result.url,'title':result.title,'status':'analyzed','score':5})
            with patch.object(main, '_analyze_one_inner', side_effect=terminal):
                main.check_watchlist()
            self.assertNotIn(oid, db.all_watchlist())

    def test_watchlist_rotation_is_bounded(self):
        import main
        with tempfile.TemporaryDirectory() as tmp, patch.object(db,'SEEN_PATH',Path(tmp)/'seen.json'), patch.object(db,'WATCHLIST_PATH',Path(tmp)/'wl.json'), patch.object(main, 'log_run_metadata'), patch.object(main.tools, 'send_notification'), patch.object(main, 'analyze_one', return_value=None) as analyze:
            for i in range(7):
                db.add_to_watchlist({'url':f'https://example.org/{i}','retry_analysis':True})
            main.check_watchlist()
            self.assertEqual(analyze.call_count,5)
            first={c.args[0].url for c in analyze.call_args_list}
            analyze.reset_mock()
            main.check_watchlist()
            self.assertTrue({'https://example.org/5','https://example.org/6'}.issubset({c.args[0].url for c in analyze.call_args_list}))

if __name__=='__main__': unittest.main()
