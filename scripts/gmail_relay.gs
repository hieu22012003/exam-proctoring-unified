/**
 * ExamWatch Gmail relay — sends 2FA login codes from your own Gmail.
 *
 * Railway blocks outbound SMTP on Trial/Hobby plans, so the backend calls this
 * Apps Script web app over HTTPS instead. Consumer Gmail allows ~100 mails/day.
 *
 * Setup (script.google.com → New project → paste this file):
 *   1. Run `setup` once (select it in the toolbar, press Run, allow access).
 *      It creates a random secret and prints it in the Execution log.
 *   2. Deploy → New deployment → type "Web app"
 *        Execute as: Me        Who has access: Anyone
 *      Copy the Web app URL (ends with /exec).
 *   3. In Railway add:
 *        EXAMWATCH_MAIL_WEBHOOK_URL    = the /exec URL
 *        EXAMWATCH_MAIL_WEBHOOK_SECRET = the secret from step 1
 *        EXAMWATCH_2FA_EMAIL           = 1
 */

function setup() {
  const props = PropertiesService.getScriptProperties();
  let secret = props.getProperty('MAIL_SECRET');
  if (!secret) {
    secret = Utilities.getUuid().replace(/-/g, '') + Utilities.getUuid().replace(/-/g, '');
    props.setProperty('MAIL_SECRET', secret);
  }
  // Touch MailApp so the authorization prompt asks for the send-mail permission.
  Logger.log('Quota còn lại hôm nay: ' + MailApp.getRemainingDailyQuota());
  Logger.log('EXAMWATCH_MAIL_WEBHOOK_SECRET = ' + secret);
}

function doPost(e) {
  try {
    const data = JSON.parse((e && e.postData && e.postData.contents) || '{}');
    const secret = PropertiesService.getScriptProperties().getProperty('MAIL_SECRET');
    if (!secret || data.secret !== secret) return reply_({ ok: false, error: 'unauthorized' });
    const to = String(data.to || '').trim();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(to)) return reply_({ ok: false, error: 'invalid recipient' });
    if (MailApp.getRemainingDailyQuota() < 1) return reply_({ ok: false, error: 'daily Gmail quota reached' });
    MailApp.sendEmail({
      to: to,
      subject: String(data.subject || 'ExamWatch').slice(0, 200),
      body: String(data.text || ''),
      htmlBody: String(data.html || ''),
      name: String(data.name || 'ExamWatch').slice(0, 60),
    });
    return reply_({ ok: true });
  } catch (err) {
    return reply_({ ok: false, error: String(err).slice(0, 200) });
  }
}

function doGet() {
  return reply_({ ok: true, service: 'examwatch-gmail-relay' });
}

function reply_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
